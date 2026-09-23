"""Detect and draw hard-coded subtitles, preserving source video timestamps."""
import argparse
from fractions import Fraction
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import av
import cv2
import imageio_ffmpeg
import numpy as np
import psutil

from subtitle_core import Record, SubtitleSegments, boxes_from_result

MODEL = 'PP-OCRv6_small_det'


def hardware():
    info = {'os': platform.platform(), 'cpu': platform.processor(),
            'logical_cpus': os.cpu_count(), 'ram_gib': round(psutil.virtual_memory().total/2**30, 2)}
    try:
        if os.name == 'nt':
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                info['cpu'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        proc = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        info['nvidia'] = proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    info['packages'] = {}
    for name in ['paddleocr', 'paddlex', 'paddlepaddle-gpu', 'paddlepaddle', 'av', 'numpy']:
        try:
            info['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return info


class Detector:
    def __init__(self, args):
        # Keep downloaded models and caches inside the workspace by default.
        root = Path(__file__).resolve().parent / '.cache'
        os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(root/'paddlex'))
        os.environ.setdefault('HF_HOME', str(root/'huggingface'))
        os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE', 'BOS')
        local_model = root/'models'/f'{args.model}_infer'
        if args.model_dir is None and local_model.is_dir():
            args.model_dir = str(local_model)
        import paddle
        from paddleocr import TextDetection
        self.paddle = paddle
        self.device = args.device
        if args.device.startswith('gpu:'):
            idx = int(args.device.split(':')[1])
            if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() <= idx:
                raise RuntimeError(f'{args.device} unavailable. Install PaddlePaddle GPU or explicitly use --device cpu.')
        elif args.device != 'cpu':
            raise ValueError('Supported devices: cpu, gpu:N')
        paddle.set_device(args.device)
        self.model = TextDetection(model_name=args.model, model_dir=args.model_dir,
                                   device=args.device, cpu_threads=args.cpu_threads,
                                   enable_mkldnn=args.enable_mkldnn, limit_side_len=args.det_size,
                                   limit_type='max', thresh=args.pixel_threshold,
                                   box_thresh=args.box_threshold, unclip_ratio=1.3)

    def sync(self):
        if self.device.startswith('gpu:'):
            self.paddle.device.cuda.synchronize()

    def predict(self, images):
        self.sync()
        start = time.perf_counter()
        results = list(self.model.predict(input=images, batch_size=len(images)))
        self.sync()
        elapsed = time.perf_counter()-start
        if len(results) != len(images):
            raise RuntimeError('Detector returned an unexpected number of results')
        return results, elapsed


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/result.mp4'))
    p.add_argument('--device', default='gpu:0')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--model-dir')
    p.add_argument('--roi-bottom', type=float, default=.45)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--det-size', type=int, default=960)
    p.add_argument('--pixel-threshold', type=float, default=.3)
    p.add_argument('--box-threshold', type=float, default=.6)
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--enable-mkldnn', action='store_true', help='Opt in to oneDNN; not compatible with every model/runtime')
    p.add_argument('--start', type=float, default=0, help='Seconds relative to the first video frame')
    p.add_argument('--duration', type=float)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--padding', type=int, default=3)
    p.add_argument('--overwrite', action='store_true')
    return p


def validate(args):
    if not args.input.is_file():
        raise ValueError(f'Input does not exist: {args.input}')
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Output must differ from input')
    if args.output.suffix.lower() != '.mp4':
        raise ValueError('Output must be .mp4')
    if not .25 <= args.roi_bottom <= .45:
        raise ValueError('--roi-bottom must be between .25 and .45')
    if args.start < 0 or (args.duration is not None and args.duration <= 0):
        raise ValueError('Start must be nonnegative and duration positive')
    if args.batch_size is None:
        args.batch_size = 8 if args.device.startswith('gpu:') else 1
    if min(args.batch_size, args.cpu_threads, args.det_size) < 1 or args.warmup < 0 or args.padding < 0:
        raise ValueError('Invalid batch size, threads, image size, warmup or padding')
    if not all(0 < x < 1 for x in [args.pixel_threshold, args.box_threshold]):
        raise ValueError('Thresholds must be between zero and one')
    for path in [args.output, args.output.with_suffix('.jsonl'), args.output.with_suffix('.metrics.json')]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f'{path} exists; use --overwrite explicitly')


def selected_frames(source, args):
    """Decode in presentation order and select the same source interval on both passes."""
    first_source_time = previous_timestamp = None
    for index, frame in enumerate(source.decode(source.streams.video[0])):
        if frame.pts is None:
            raise RuntimeError('Source frame missing PTS; cannot guarantee audio synchronization')
        timestamp = float(frame.pts*frame.time_base)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise RuntimeError('Source timestamps must increase strictly')
        previous_timestamp = timestamp
        if first_source_time is None:
            first_source_time = timestamp
        relative = timestamp-first_source_time
        if relative < args.start-1e-7:
            continue
        if args.duration is not None and relative >= args.start+args.duration-1e-7:
            break
        yield index, timestamp, relative, frame


def run(args, detector=None):
    validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    init_start = time.perf_counter()
    if detector is None:
        print(f'Loading {args.model} on {args.device}...', flush=True)
        detector = Detector(args)
        print('Model ready.', flush=True)
    init_seconds = time.perf_counter()-init_start
    metrics = {'model': args.model, 'device': args.device, 'hardware': hardware(),
               'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
               'initialization_seconds': init_seconds, 'warmup_seconds': 0,
               'inference_seconds': 0., 'postprocess_seconds': 0., 'encode_seconds': 0.,
               'decode_seconds': 0., 'batch_latency_ms': [], 'frames': 0,
               'timing_note': 'Inference includes PaddleX preprocessing and detection postprocessing; latency is per batch.'}
    with av.open(str(args.input)) as source, tempfile.TemporaryDirectory(prefix='subtitle-', dir=args.output.parent) as tmp:
        if not source.streams.video:
            raise ValueError('Input has no video stream')
        stream = source.streams.video[0]
        width, height = stream.width, stream.height
        rate = stream.average_rate or Fraction(30)
        tb = stream.time_base
        roi_y = int(height*(1-args.roi_bottom))
        metrics['video'] = {'width': width, 'height': height, 'fps': float(rate), 'time_base': str(tb)}
        video_path = Path(tmp)/'silent.mp4'
        json_path = Path(tmp)/'boxes.jsonl'
        final_path = Path(tmp)/'result.mp4'
        segments = SubtitleSegments(width, height, args.padding)
        frames, batch = [], []
        process_start = time.perf_counter()
        warmed = False

        def infer_batch():
            nonlocal warmed
            crops = [record.image[roi_y:].copy() for record, _ in batch]
            if not warmed:
                begin = time.perf_counter()
                for _ in range(args.warmup):
                    detector.predict(crops)
                metrics['warmup_seconds'] = time.perf_counter()-begin
                warmed = True
            results, elapsed = detector.predict(crops)
            metrics['inference_seconds'] += elapsed
            metrics['batch_latency_ms'].append(1000*elapsed)
            begin = time.perf_counter()
            for (record, relative), result in zip(batch, results):
                boxes = boxes_from_result(result, roi_y, width, height, args.box_threshold)
                segments.add(record.gray, boxes)
                frames.append((record.index, record.timestamp, relative, record.duration))
            metrics['postprocess_seconds'] += time.perf_counter()-begin
            batch.clear()
            if len(frames) and len(frames) % 240 < args.batch_size:
                print(f'Detected {len(frames)} frames', flush=True)

        # Pass 1: detect every selected frame, then settle each subtitle run's box.
        decoded = iter(selected_frames(source, args))
        while True:
            begin = time.perf_counter()
            try:
                index, timestamp, relative, frame = next(decoded)
            except StopIteration:
                break
            image = frame.to_ndarray(format='bgr24')
            duration = float(frame.duration*frame.time_base) if frame.duration else 1/float(rate)
            batch.append((Record(index, timestamp, duration, image,
                                 cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), []), relative))
            metrics['decode_seconds'] += time.perf_counter()-begin
            if len(batch) == args.batch_size:
                infer_batch()
        if batch:
            infer_batch()
        if not frames:
            raise ValueError('Selected video interval has no frames')
        begin = time.perf_counter()
        stable_boxes = segments.finish()
        metrics['postprocess_seconds'] += time.perf_counter()-begin
        metrics['subtitle_segments'] = segments.count
        first_output_time = frames[0][1]
        last_time = frames[-1][1]
        last_duration = frames[-1][3]
        video_duration = last_time-first_output_time+last_duration

        # Pass 2: draw the finalized boxes on the original frames and preserve PTS.
        with av.open(str(args.input)) as original, av.open(str(video_path), 'w') as sink, \
                json_path.open('w', encoding='utf-8') as records:
            enc = sink.add_stream('libx264', rate=rate)
            enc.width, enc.height = width, height
            enc.pix_fmt = 'yuv420p' if width%2 == 0 and height%2 == 0 else 'yuv444p'
            enc.time_base = tb
            enc.codec_context.time_base = tb
            enc.options = {'crf': '18', 'preset': 'fast', 'bf': '0'}
            for output_index, (index, timestamp, _, frame) in enumerate(selected_frames(original, args)):
                if output_index >= len(frames):
                    raise RuntimeError('Source changed between detection and rendering')
                expected_index, expected_time, expected_relative, duration = frames[output_index]
                if index != expected_index or abs(timestamp-expected_time) > 1e-7:
                    raise RuntimeError('Source changed between detection and rendering')
                begin = time.perf_counter()
                image = frame.to_ndarray(format='bgr24')
                metrics['decode_seconds'] += time.perf_counter()-begin
                begin = time.perf_counter()
                drawn = stable_boxes[output_index]
                for x1, y1, x2, y2 in drawn:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                records.write(json.dumps({'frame_index': index,
                                          'source_timestamp_seconds': expected_relative,
                                          'output_timestamp_seconds': timestamp-first_output_time,
                                          'boxes_xyxy': drawn})+'\n')
                metrics['postprocess_seconds'] += time.perf_counter()-begin
                begin = time.perf_counter()
                output_frame = av.VideoFrame.from_ndarray(image, format='bgr24')
                output_frame.pts = round((timestamp-first_output_time)/float(tb))
                output_frame.time_base = tb
                output_frame.duration = max(1, round(duration/float(tb)))
                for packet in enc.encode(output_frame):
                    sink.mux(packet)
                metrics['encode_seconds'] += time.perf_counter()-begin
                metrics['frames'] += 1
            if metrics['frames'] != len(frames):
                raise RuntimeError('Source changed between detection and rendering')
            begin = time.perf_counter()
            for packet in enc.encode():
                sink.mux(packet)
            metrics['encode_seconds'] += time.perf_counter()-begin

        mux_start = time.perf_counter()
        # Audio trim uses source presentation timestamps, including stream offsets.
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
               '-i', str(video_path)]
        if source.streams.audio:
            cmd += ['-copyts', '-i', str(args.input), '-filter_complex',
                    f'[1:a:0]atrim=start={first_output_time:.9f}:end={first_output_time+video_duration:.9f},'
                    f'asetpts=PTS-({first_output_time:.9f})/TB[a]',
                    '-map', '0:v:0', '-map', '[a]', '-c:a', 'aac', '-b:a', '192k']
        else:
            cmd += ['-map', '0:v:0', '-an']
        cmd += ['-c:v', 'copy', '-movflags', '+faststart', str(final_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f'FFmpeg mux failed: {proc.stderr}')
        metrics['mux_seconds'] = time.perf_counter()-mux_start
        metrics['processing_seconds'] = time.perf_counter()-process_start-metrics['warmup_seconds']
        metrics['video']['processed_duration_seconds'] = video_duration
        metrics['effective_fps'] = metrics['frames']/metrics['processing_seconds']
        latencies = metrics.pop('batch_latency_ms')
        metrics['batch_latency_ms_median'] = float(np.median(latencies))
        metrics['batch_latency_ms_p95'] = float(np.percentile(latencies,95))
        metrics['amortized_inference_ms_per_frame'] = metrics['inference_seconds']/metrics['frames']*1000
        # Decode the finished file before publishing it.
        verify_start = time.perf_counter()
        with av.open(str(final_path)) as check, json_path.open(encoding='utf-8') as expected:
            count = 0
            for f in check.decode(video=0):
                if (f.width, f.height) != (width, height):
                    raise RuntimeError('Output dimensions changed')
                line = expected.readline()
                if not line:
                    raise RuntimeError('Output contains extra frames')
                timestamp = json.loads(line)['output_timestamp_seconds']
                if f.pts is None or abs(float(f.pts*f.time_base)-timestamp) > max(float(tb),float(f.time_base))*1.1:
                    raise RuntimeError('Output timestamp mismatch')
                count += 1
            if count != metrics['frames']:
                raise RuntimeError(f'Output frame count mismatch: {count}')
            actual_duration = float(check.streams.video[0].duration*check.streams.video[0].time_base)
            if abs(actual_duration-video_duration) > max(last_duration,1/float(rate))+1e-5:
                raise RuntimeError('Output video duration changed by more than one frame')
            if bool(check.streams.audio) != bool(source.streams.audio):
                raise RuntimeError('Output audio stream missing or unexpected')
        metrics['verification_seconds'] = time.perf_counter()-verify_start
        metrics['verified_frame_count'] = count
        metrics['total_seconds_including_init_warmup_verify'] = time.perf_counter()-init_start
        final_path.replace(args.output)
        json_path.replace(args.output.with_suffix('.jsonl'))
    args.output.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Saved {args.output}: {metrics['frames']} frames, {metrics['processing_seconds']:.2f}s, {metrics['effective_fps']:.1f} FPS", flush=True)
    return metrics


if __name__ == '__main__':
    try:
        run(parser().parse_args())
    except (ValueError, RuntimeError, FileExistsError) as exc:
        raise SystemExit(str(exc))

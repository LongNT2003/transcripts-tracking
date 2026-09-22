"""Real codec/mux integration tests; detector stub isolates video correctness."""
from fractions import Fraction
import json
import subprocess
import av
import imageio_ffmpeg
import numpy as np
import pytest
from detect_subtitles import parser, run


class StubDetector:
    def predict(self, images):
        return [{'dt_polys': [], 'dt_scores': []} for _ in images], .001


def make_video(path, times):
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('libx264', rate=30)
        stream.width, stream.height = 160, 120
        stream.pix_fmt = 'yuv420p'
        stream.time_base = Fraction(1, 90000)
        stream.codec_context.time_base = stream.time_base
        stream.options = {'bf': '0'}
        for i,t in enumerate(times):
            image = np.full((120,160,3), (i*15)%256, np.uint8)
            frame = av.VideoFrame.from_ndarray(image,format='bgr24')
            frame.pts = round(t*90000)
            frame.time_base = Fraction(1,90000)
            for pkt in stream.encode(frame):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)


def timestamps(path):
    with av.open(str(path)) as source:
        return [float(f.pts*f.time_base) for f in source.decode(video=0)]


@pytest.mark.parametrize('vfr', [False, True])
@pytest.mark.parametrize('audio', [False, True])
def test_video_timestamps_and_audio(tmp_path, vfr, audio):
    path = tmp_path/'input.mp4'
    times = [0,.033333,.1,.133333,.233333,.3] if vfr else [i/30 for i in range(9)]
    make_video(path,times)
    if audio:
        with_audio = tmp_path/'audio.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel','error','-i',str(path),
                        '-f','lavfi','-i','sine=frequency=500:sample_rate=48000:duration=1',
                        '-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-shortest',str(with_audio)],check=True)
        path = with_audio
    output = tmp_path/'output.mp4'
    args = parser().parse_args(['--input',str(path),'--output',str(output),'--device','cpu','--warmup','0'])
    metrics = run(args,detector=StubDetector())
    assert np.allclose(timestamps(path),timestamps(output),atol=1/90000)
    assert metrics['verified_frame_count'] == len(times)
    with av.open(str(output)) as source:
        assert bool(source.streams.audio) == audio
        video = source.streams.video[0]
        if audio:
            sound = source.streams.audio[0]
            with av.open(str(path)) as original:
                original_sound = original.streams.audio[0]
                target = min(float(original_sound.duration*original_sound.time_base), float(video.duration*video.time_base))
                assert abs(float(sound.duration*sound.time_base)-target) < 1024/48000
    assert len(output.with_suffix('.jsonl').read_text().splitlines()) == len(times)


def test_trim_and_output_protection(tmp_path):
    path, output = tmp_path/'input.mp4', tmp_path/'output.mp4'
    make_video(path, [i/30 for i in range(15)])
    args = parser().parse_args(['--input',str(path),'--output',str(output),'--device','cpu',
                                '--start','0.1','--duration','0.2','--warmup','0'])
    run(args,detector=StubDetector())
    rows = [json.loads(line) for line in output.with_suffix('.jsonl').read_text().splitlines()]
    assert len(rows) == 6
    assert rows[0]['source_timestamp_seconds'] == pytest.approx(.1)
    assert rows[0]['output_timestamp_seconds'] == 0
    with pytest.raises(FileExistsError):
        run(args,detector=StubDetector())


def test_delayed_audio_keeps_offset_after_trim(tmp_path):
    silent, path, output = tmp_path/'silent.mp4', tmp_path/'delayed.mp4', tmp_path/'out.mp4'
    make_video(silent, [i/30 for i in range(30)])
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel','error','-i',str(silent),
                    '-itsoffset','0.3','-f','lavfi','-i','sine=frequency=500:sample_rate=48000:duration=0.5',
                    '-map','0:v','-map','1:a','-c:v','copy','-c:a','aac',str(path)],check=True)
    args = parser().parse_args(['--input',str(path),'--output',str(output),'--device','cpu',
                                '--start','0.1','--duration','0.7','--warmup','0'])
    run(args,detector=StubDetector())
    with av.open(str(path)) as source, av.open(str(output)) as result:
        a, b = source.streams.audio[0], result.streams.audio[0]
        # One AAC frame of encoder priming is tolerated (1024 samples at 48 kHz).
        assert abs(float(b.start_time*b.time_base) - (float(a.start_time*a.time_base)-.1)) < .025

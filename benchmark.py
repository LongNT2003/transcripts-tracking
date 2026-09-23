"""Repeat the actual video pipeline and report measured and evidence-backed estimates."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

HARDWARE = ['Core i5/i7 CPU, 16 GB RAM', 'RTX 3050', 'RTX 3060',
            'RTX 4070', 'RTX 4090', 'Tesla T4', 'Apple M1 Pro/Max',
            'Apple M2 Pro/Max', 'Apple M3 Pro/Max']


def build_report(directory, estimates=None):
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('run-*.metrics.json'))]
    groups = {}
    for row in runs:
        groups.setdefault(row['device'], []).append(row)
    summary = []
    for device, values in groups.items():
        summary.append({'device': device, 'type': 'measured', 'runs': len(values),
                        'seconds_median': statistics.median(v['processing_seconds'] for v in values),
                        'seconds_min': min(v['processing_seconds'] for v in values),
                        'seconds_max': max(v['processing_seconds'] for v in values),
                        'fps_median': statistics.median(v['effective_fps'] for v in values),
                        'inference_ms_per_frame': statistics.median(v['amortized_inference_ms_per_frame'] for v in values)})
    with (directory/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['device','type','runs','seconds_median','seconds_min','seconds_max','fps_median','inference_ms_per_frame'])
        writer.writeheader()
        writer.writerows(summary)
    # External estimates must use the same model/ROI/backend and include evidence.
    supplied = json.loads(estimates.read_text(encoding='utf-8')) if estimates else []
    for entry in supplied:
        for field in ['hardware','model','backend','source','assumptions','seconds_low','seconds_high','video_duration_seconds']:
            if field not in entry or entry[field] in ('', None):
                raise ValueError(f'Estimate missing {field}')
        if not 0 < entry['seconds_low'] <= entry['seconds_high'] or entry['video_duration_seconds'] <= 0:
            raise ValueError('Invalid estimate interval')
    lines = ['# Bao cao thoi gian xu ly', '',
             'So lieu do thuc te bao gom decode, detect, on dinh box, encode va ghep audio. '
             'Khoi tao/download model, warm-up va decode kiem tra output duoc ghi rieng trong JSON.', '',
             '| Device | Runs | Median seconds | Min–max | FPS | Inference ms/frame |',
             '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['device']} | {r['runs']} | {r['seconds_median']:.2f} | {r['seconds_min']:.2f}–{r['seconds_max']:.2f} | {r['fps_median']:.2f} | {r['inference_ms_per_frame']:.2f} |")
    if runs:
        lines += ['', '## Cau hinh va doan video', '', '```json', json.dumps({
            'hardware': runs[0]['hardware'], 'video': runs[0]['video'], 'config': runs[0]['config']}, ensure_ascii=False, indent=2), '```']
    lines += ['', '## Uoc luong phan cung khac', '',
              'Khong suy toc do tu TFLOPS, ten GPU hay VRAM. Can benchmark cung model, ROI, '
              'kich thuoc inference va backend; ket qua Apple CPU khong dai dien cho Metal/ANE.', '',
              '| Hardware | Status | Time range (seconds) | Evidence / assumptions |', '|---|---|---|---|']
    for name in HARDWARE:
        matches = [e for e in supplied if e['hardware'] == name]
        if matches:
            e = matches[0]
            detail = f"{e['model']}; {e['backend']}; video {e['video_duration_seconds']}s; {e['source']}; {e['assumptions']}"
            lines.append(f"| {name} | estimated | {e['seconds_low']}–{e['seconds_high']} | {detail.replace('|','/')} |")
        else:
            lines.append(f'| {name} | Chua du du lieu | — | Chua co benchmark tuong duong |')
    (directory/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (directory/'summary.json').write_text(json.dumps({'measured': summary, 'estimates': supplied}, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--devices', nargs='+', default=['cpu','gpu:0'])
    p.add_argument('--duration', type=float, default=30)
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--batch-size', type=int, help='Frames per inference call; default follows device (CPU 1, GPU 8)')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/benchmark'))
    p.add_argument('--estimates', type=Path)
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if args.repeats < 1 or args.duration <= 0:
        p.error('Repeats and duration must be positive')
    if args.batch_size is not None and args.batch_size < 1:
        p.error('Batch size must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = args.output_dir/'failures.json'
    failures = json.loads(failure_path.read_text(encoding='utf-8')) if args.report_only and failure_path.exists() else []
    if not args.report_only:
        if list(args.output_dir.glob('run-*.metrics.json')):
            p.error('Benchmark directory already contains runs; choose a new directory or --report-only')
        for device in args.devices:
            for repeat in range(args.repeats):
                output = args.output_dir/f"run-{device.replace(':','_')}-{repeat+1}.mp4"
                command = [sys.executable, str(Path(__file__).with_name('detect_subtitles.py')),
                           '--input', str(args.input.resolve()), '--output', str(output.resolve()),
                           '--device', device, '--start', str(args.start), '--duration', str(args.duration)]
                if args.batch_size is not None:
                    command.extend(['--batch-size', str(args.batch_size)])
                with output.with_suffix('.log').open('w', encoding='utf-8') as log:
                    proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if proc.returncode:
                    failures.append({'device': device, 'repeat': repeat+1, 'log': str(output.with_suffix('.log'))})
                    print(f'Failed {device}; see {failures[-1]["log"]}', flush=True)
                    break
                print(f'Completed {device} repeat {repeat+1}', flush=True)
    failure_path.write_text(json.dumps(failures, indent=2), encoding='utf-8')
    build_report(args.output_dir, args.estimates)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

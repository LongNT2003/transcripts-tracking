import json
import pytest
from benchmark import build_report


def test_report_does_not_fabricate_remote_results(tmp_path):
    build_report(tmp_path)
    assert 'Chua du du lieu' in (tmp_path/'report.md').read_text(encoding='utf-8')
    assert json.loads((tmp_path/'summary.json').read_text()) == {'measured': [], 'estimates': []}


def test_estimate_requires_evidence(tmp_path):
    path = tmp_path/'estimates.json'
    path.write_text('[{"hardware":"Tesla T4","seconds_low":1,"seconds_high":2}]')
    with pytest.raises(ValueError,match='missing'):
        build_report(tmp_path,path)


@pytest.mark.parametrize('batch_size', [None, 4])
def test_benchmark_passes_optional_batch_size_to_detector(tmp_path, monkeypatch, batch_size):
    import sys
    import benchmark

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type('Result', (), {'returncode': 0})()

    monkeypatch.setattr(benchmark.subprocess, 'run', fake_run)
    argv = ['benchmark.py', '--input', str(tmp_path/'input.mp4'),
            '--devices', 'cpu', '--repeats', '1', '--output-dir', str(tmp_path/'runs')]
    if batch_size is not None:
        argv.extend(['--batch-size', str(batch_size)])
    monkeypatch.setattr(sys, 'argv', argv)
    benchmark.main()
    assert len(calls) == 1
    if batch_size is None:
        assert '--batch-size' not in calls[0]
    else:
        position = calls[0].index('--batch-size')
        assert calls[0][position + 1] == str(batch_size)


def test_benchmark_rejects_nonpositive_batch_size(tmp_path, monkeypatch):
    import sys
    import benchmark

    monkeypatch.setattr(sys, 'argv', ['benchmark.py', '--input', 'video.mp4',
                                     '--output-dir', str(tmp_path/'runs'), '--batch-size', '0'])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2

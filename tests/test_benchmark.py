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

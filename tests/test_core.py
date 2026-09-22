import cv2
import numpy as np
import pytest
from subtitle_core import Record, Stabilizer, boxes_from_result, merge_lines


def record(boxes, text='ABC', background=0):
    img = np.full((120, 240), background, np.uint8)
    if text:
        cv2.putText(img, text, (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return Record(0,0,1/30,cv2.cvtColor(img,cv2.COLOR_GRAY2BGR),img,boxes)


BOX = [30,60,110,96]


def test_roi_mapping_and_confidence():
    result = {'dt_polys': [[[10,5],[100,5],[100,25],[10,25]], [[0,0],[50,0],[50,20],[0,20]]],
              'dt_scores': [.9,.1]}
    assert boxes_from_result(result, 550, 720, 1000) == [[10,555,100,575]]


def test_merge_fragments_not_lines():
    boxes = [[10,50,40,70], [45,51,100,71], [10,80,100,100]]
    assert merge_lines(boxes) == [[10,50,100,71], [10,80,100,100]]


@pytest.mark.parametrize('gap', [1,2])
def test_bridge_short_detection_gap(gap):
    s = Stabilizer()
    s.step(record([BOX]), [])
    for i in range(gap):
        future = [record([]) for _ in range(gap-i-1)] + [record([BOX])]
        assert s.step(record([]), future) == [BOX]


def test_does_not_hold_box_after_subtitle_disappears():
    s = Stabilizer()
    s.step(record([BOX]), [])
    assert s.step(record([],text=''), [record([BOX])]) == []


def test_no_bridge_without_future_detection():
    s = Stabilizer()
    s.step(record([BOX]), [])
    assert s.step(record([]), []) == []


def test_changed_line_snaps_to_new_width():
    s = Stabilizer()
    s.step(record([BOX]), [])
    wider = [30,60,210,96]
    assert s.step(record([wider],text='ABCDEFG'), []) == [wider]


def test_scene_cut_clears_previous_tracks():
    s = Stabilizer()
    s.step(record([BOX]), [])
    assert s.step(record([],text='',background=240), [record([BOX])]) == []

"""Geometry and bounded, two-frame lookahead subtitle stabilization."""
from dataclasses import dataclass
import cv2
import numpy as np


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    return intersection / max(1, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def merge_lines(boxes):
    lines = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        box = list(box)
        merged = True
        while merged:
            merged = False
            for j, other in enumerate(lines):
                h = min(box[3]-box[1], other[3]-other[1])
                overlap = min(box[3], other[3]) - max(box[1], other[1])
                gap = max(box[0], other[0]) - min(box[2], other[2])
                if overlap >= .65*h and gap <= .9*h:
                    box = [min(box[0], other[0]), min(box[1], other[1]),
                           max(box[2], other[2]), max(box[3], other[3])]
                    lines.pop(j)
                    merged = True
                    break
        lines.append(box)
    return sorted(lines, key=lambda b: (b[1], b[0]))


def boxes_from_result(result, offset_y, width, height, confidence=.6):
    """Paddle coordinates are relative to the unscaled ROI input."""
    boxes = []
    for poly, score in zip(result['dt_polys'], result['dt_scores']):
        if float(score) < confidence:
            continue
        p = np.asarray(poly, dtype=float)
        if p.shape != (4, 2) or not np.isfinite(p).all():
            continue
        # Reject rotated signage; allow modest skew in subtitle glyph outlines.
        edge = p[1]-p[0]
        if abs(edge[1]) > .25*max(1, abs(edge[0])):
            continue
        x1, y1 = p.min(axis=0)
        x2, y2 = p.max(axis=0)
        y1 += offset_y
        y2 += offset_y
        bw, bh = x2-x1, y2-y1
        if not (.009*height <= bh <= .085*height and bw >= .6*bh):
            continue
        boxes.append([max(0, x1), max(offset_y, y1), min(width, x2), min(height, y2)])
    return merge_lines(boxes)


def appearance(gray, box):
    x1, y1, x2, y2 = np.rint(box).astype(int)
    h, w = gray.shape
    crop = gray[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if crop.size == 0:
        return np.zeros((24, 160), np.float32)
    resized = cv2.resize(crop, (160, 24))
    # Gradient signature emphasizes glyphs over slowly changing backgrounds.
    return cv2.Canny(resized, 70, 160).astype(np.float32)/255


def similar(a, b, threshold=.55):
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return bool(denom > 1 and float(np.sum(a*b)/denom) >= threshold)


@dataclass
class Record:
    index: int
    timestamp: float
    duration: float
    image: np.ndarray
    gray: np.ndarray
    boxes: list


class Stabilizer:
    """Only bridge a gap if a future detection and current glyphs support it."""
    def __init__(self, alpha=.45, max_gap=2):
        self.alpha = alpha
        self.max_gap = max_gap
        self.tracks = []
        self.previous_scene = None

    def step(self, current, future):
        scene = cv2.resize(current.gray, (48, 48)).astype(np.float32)
        if self.previous_scene is not None and np.mean(abs(scene-self.previous_scene)) > 45:
            self.tracks = []
        self.previous_scene = scene
        remaining = set(range(len(self.tracks)))
        new_tracks, result = [], []
        for box in current.boxes:
            candidate = max(remaining, key=lambda j: iou(box, self.tracks[j][0]), default=None)
            smooth = np.asarray(box, float)
            if candidate is not None and iou(box, self.tracks[candidate][0]) >= .65:
                old, _, _ = self.tracks[candidate]
                if similar(appearance(current.gray, old), self.tracks[candidate][1]):
                    smooth = self.alpha*smooth + (1-self.alpha)*np.asarray(old)
                    # Bound smoothing lag so a box cannot clip a changed line.
                    smooth = np.clip(smooth, np.asarray(box)-2, np.asarray(box)+2)
                remaining.remove(candidate)
            signature = appearance(current.gray, smooth)
            new_tracks.append((smooth.tolist(), signature, 0))
            result.append(smooth.tolist())
        for j in remaining:
            box, sig, missed = self.tracks[j]
            if missed >= self.max_gap or any(iou(box, b) > .2 for b in current.boxes):
                continue
            if not similar(sig, appearance(current.gray, box)):
                continue
            supported = any(
                any(iou(box, b) >= .65 for b in f.boxes)
                and similar(sig, appearance(f.gray, box))
                for f in future[:self.max_gap-missed]
            )
            if supported:
                result.append(box)
                new_tracks.append((box, sig, missed+1))
        self.tracks = new_tracks
        return sorted(result, key=lambda b: (b[1], b[0]))


class SubtitleSegments:
    """Assign one robust, screen-space box to each sufficiently supported run."""

    def __init__(self, width, height, padding=3, max_gap=2,
                 min_seconds=.8, min_detection_frames=3):
        self.width, self.height = width, height
        self.padding, self.max_gap = padding, max_gap
        self.min_seconds = min_seconds
        self.min_detection_frames = min_detection_frames
        self.frames = []
        self.active = []
        self.completed = []
        self.count = 0
        self.discarded_short_segments = 0

    def _assign(self, track):
        coords = np.asarray(track['boxes'], dtype=float)
        low = np.quantile(coords[:, :2], .1, axis=0)
        high = np.quantile(coords[:, 2:], .9, axis=0)
        x1, y1 = np.floor(low).astype(int) - self.padding
        x2, y2 = np.ceil(high).astype(int) + self.padding
        box = [max(0, int(x1)), max(0, int(y1)),
               min(self.width-1, int(x2)), min(self.height-1, int(y2))]
        if box[0] >= box[2] or box[1] >= box[3]:
            return
        for index in range(track['start'], track['last']+1):
            self.frames[index].append(box)
        self.count += 1

    def add(self, gray, boxes, timestamp, duration):
        index = len(self.frames)
        self.frames.append([])
        signatures = [appearance(gray, track['anchor']) for track in self.active]
        candidates = []
        for ti, track in enumerate(self.active):
            for bi, box in enumerate(boxes):
                overlap = iou(track['boxes'][-1], box)
                if overlap >= .5 and similar(track['signature'], signatures[ti], threshold=.8):
                    candidates.append((overlap, ti, bi))
        matched_tracks, matched_boxes = set(), set()
        for _, ti, bi in sorted(candidates, reverse=True):
            if ti in matched_tracks or bi in matched_boxes:
                continue
            track = self.active[ti]
            track['boxes'].append(boxes[bi])
            track['signature'] = signatures[ti]
            track['last'] = index
            track['end_time'] = timestamp + duration
            track['detections'] += 1
            matched_tracks.add(ti)
            matched_boxes.add(bi)
        remaining = []
        for ti, track in enumerate(self.active):
            if ti in matched_tracks:
                remaining.append(track)
            elif (index-track['last'] <= self.max_gap
                  and not any(iou(track['boxes'][-1], box) > .2 for box in boxes)
                  and similar(track['signature'], signatures[ti], threshold=.8)):
                remaining.append(track)
            else:
                self.completed.append(track)
        self.active = remaining
        for bi, box in enumerate(boxes):
            if bi not in matched_boxes:
                self.active.append({'start': index, 'last': index, 'anchor': box,
                                    'boxes': [box], 'signature': appearance(gray, box),
                                    'start_time': timestamp, 'end_time': timestamp+duration,
                                    'detections': 1})

    def finish(self, preserve_first=False, preserve_last=False):
        self.completed.extend(self.active)
        self.active.clear()
        last_frame = len(self.frames)-1
        for track in self.completed:
            at_trim_edge = ((preserve_first and track['start'] == 0)
                            or (preserve_last and track['last'] == last_frame))
            span = track['end_time']-track['start_time']
            if not at_trim_edge and (track['detections'] < self.min_detection_frames
                                     or span + 1e-9 < self.min_seconds):
                self.discarded_short_segments += 1
                continue
            self._assign(track)
        self.completed.clear()
        for boxes in self.frames:
            boxes.sort(key=lambda box: (box[1], box[0]))
        return self.frames

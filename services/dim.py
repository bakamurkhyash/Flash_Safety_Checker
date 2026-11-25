
import argparse, csv
from collections import deque

import cv2
import numpy as np
import cv2


def dim():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--factor", type=float, default=0.5, help="fixed brightness multiplier on flash frames")
    ap.add_argument("--auto", action="store_true",
                    help="per-frame gain that pulls the frame back to the pre-flash baseline luminance")
    ap.add_argument("--pad", type=int, default=0, help="also dim N frames before/after each flash")
    ap.add_argument("--window", type=float, default=1.0, help="baseline window in seconds (--auto)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)#
    if not cap.isOpened():
        raise SystemExit("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frame mask from CSV (O(1) lookup per frame)
    mask = np.zeros(total + 1, dtype=bool)
    with open(a.csv) as f:
        for r in csv.DictReader(f):
            s = max(0, int(r["start_frame"]) - a.pad)
            e = min(total, int(r["end_frame"]) + a.pad)
            mask[s:e + 1] = True

    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    lut = np.clip(np.arange(256) * a.factor, 0, 255).astype(np.uint8)  # fixed-factor lookup table
    baseline = deque(maxlen=max(5, int(a.window * fps)))
    small_w = 160
    small_size = (small_w, max(1, int(small_w * h / w)))

    idx = dimmed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        elif idx <= total and mask[idx]:
            if a.auto and baseline:
                small = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA)
                lum = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
                gain = min(1.0, float(np.median(baseline)) / max(lum, 1e-3))
                frame = cv2.convertScaleAbs(frame, alpha=gain)
            else:
                frame = cv2.LUT(frame, lut)
            dimmed += 1
        elif a.auto:  # only non-flash frames feed the baseline
            small = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA)
            baseline.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()))

        out.write(frame)
        idx += 1

    cap.release()
    out.release()
    print(f"Done: {dimmed} of {idx} frames dimmed -> {a.out}")
    print(f"Add audio: ffmpeg -i {a.out} -i {a.video} -map 0:v -map 1:a? -c copy final.mp4")


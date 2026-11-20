import argparse, csv
from collections import deque

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--thresh", type=float, default=25.0,
                    help="min rise (0-255 luminance) above rolling baseline = flash")
    ap.add_argument("--width", type=int, default=160, help="downscale width for analysis")
    ap.add_argument("--stride", type=int, default=1,
                    help="analyse every Nth frame (1 = all; keep 1 for single-frame flashes)")
    ap.add_argument("--window", type=float, default=1.0, help="baseline window in seconds")
    ap.add_argument("--out", default="flashes.csv")
    ap.add_argument("--save-frames", action="store_true", help="save peak frame of each flash as JPG")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h0 = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (a.width, max(1, int(a.width * h0 / w0)))

    baseline = deque(maxlen=max(5, int(a.window * fps / a.stride)))
    events, cur = [], None   # cur: dict for the flash in progress
    idx = 0

    while True:
        if a.stride > 1 and idx % a.stride:
            if not cap.grab():          # skip without color conversion/retrieve
                break
            idx += 1
            continue

        ok, frame = cap.read()
        if not ok:
            break

        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        lum = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())

        flash = False
        if len(baseline) >= baseline.maxlen // 2:
            flash = (lum - float(np.median(baseline))) >= a.thresh

        if flash:
            if cur is None:
                cur = dict(start_frame=idx, start_time=idx / fps,
                           peak_frame=idx, peak_lum=lum, peak_img=frame.copy() if a.save_frames else None)
            if lum > cur["peak_lum"]:
                cur.update(peak_frame=idx, peak_lum=lum)
                if a.save_frames:
                    cur["peak_img"] = frame.copy()
            cur["end_frame"], cur["end_time"] = idx, idx / fps
        else:
            baseline.append(lum)        # keep flashes out of the baseline
            if cur is not None:
                events.append(cur)
                cur = None
        idx += 1

    if cur is not None:
        events.append(cur)
    cap.release()

    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["start_frame", "end_frame", "peak_frame", "start_time_s",
                     "end_time_s", "peak_time_s", "peak_luminance"])
        for i, e in enumerate(events):
            wr.writerow([e["start_frame"], e["end_frame"], e["peak_frame"],
                         f"{e['start_time']:.3f}", f"{e['end_time']:.3f}",
                         f"{e['peak_frame'] / fps:.3f}", f"{e['peak_lum']:.1f}"])
            if a.save_frames and e["peak_img"] is not None:
                cv2.imwrite(f"flash_{i:03d}_f{e['peak_frame']}.jpg", e["peak_img"])
            print(f"flash {i}: frames {e['start_frame']}-{e['end_frame']} "
                  f"(peak {e['peak_frame']}) @ {e['peak_frame'] / fps:.3f}s  lum={e['peak_lum']:.1f}")

    print(f"\n{len(events)} flash event(s) in {idx}/{total} frames -> {a.out}")


if __name__ == "__main__":
    main()

"""
Click court landmarks once to calibrate a fixed camera.

    python tools/calibrate.py game.mp4 -o configs/my_gym_calibration.json [--frame 120]

For each landmark name shown, left-click it in the image, or press:
    s   skip (landmark not visible)
    u   undo the last point
    q   finish (needs at least 4 points; 6+ spread across the court is better)

The JSON is then used with `court.calibration` in the config. It suits fixed
cameras (gym, phone on a tripod). Moving broadcast cameras need a court
keypoint model instead; see the README.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from courtvision.court import LANDMARKS, Minimap, estimate_homography, to_court  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read frame {args.frame} of {args.video}")

    names = list(LANDMARKS)
    points = {}
    state = {"idx": 0, "click": None}
    minimap = Minimap(420)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", on_mouse)

    while state["idx"] < len(names):
        name = names[state["idx"]]
        view = frame.copy()
        for n, (x, y) in points.items():
            cv2.circle(view, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(view, n, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(view, f"Click: {name}   [s]kip [u]ndo [q]uit  ({len(points)} pts)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Show where this landmark is on the court diagram
        mm = minimap.base.copy()
        cv2.circle(mm, minimap.px(*LANDMARKS[name]), 7, (0, 255, 255), -1)
        cv2.imshow("landmark", mm)
        cv2.imshow("calibrate", view)

        key = cv2.waitKey(30) & 0xFF
        if state["click"] is not None:
            points[name] = list(state["click"])
            state["click"] = None
            state["idx"] += 1
        elif key == ord("s"):
            state["idx"] += 1
        elif key == ord("u") and points:
            last = list(points)[-1]
            del points[last]
            state["idx"] = names.index(last)
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()

    if len(points) < 4:
        sys.exit(f"Only {len(points)} points; need at least 4.")

    img = np.array(list(points.values()), dtype=np.float32)
    court = np.array([LANDMARKS[n] for n in points], dtype=np.float32)
    H = estimate_homography(img, court)
    err = np.linalg.norm(to_court(H, img) - court, axis=1)
    print(f"Reprojection error: mean {err.mean():.2f} m, max {err.max():.2f} m")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"video": os.path.basename(args.video), "frame": args.frame,
                   "image_size": [frame.shape[1], frame.shape[0]], "points": points}, f, indent=2)
    print(f"Saved {len(points)} landmarks to {args.out}")


if __name__ == "__main__":
    main()

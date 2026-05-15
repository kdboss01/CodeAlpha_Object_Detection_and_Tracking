"""
Task 4: Real-Time Object Detection and Tracking
================================================
Stack:  YOLOv8 (Ultralytics) + Deep SORT (deep-sort-realtime)
Input:  Webcam (default) or video file path via --source argument
Output: Live annotated window + optional saved video

Usage:
    python detect_and_track.py                        # webcam
    python detect_and_track.py --source video.mp4     # video file
    python detect_and_track.py --source video.mp4 --save  # save output
"""

import argparse
import time
import cv2
import numpy as np

# ─────────────────────────────────────────────
#  Lazy imports so we give helpful errors early
# ─────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Install ultralytics: pip install ultralytics")

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    raise ImportError("Install tracker: pip install deep-sort-realtime")


# ─────────────────────────────────────────────
#  Config — tweak these without touching logic
# ─────────────────────────────────────────────
YOLO_MODEL      = "yolov8s.pt"   # n=nano(fast), s=small, m=medium, l=large
CONF_THRESHOLD  = 0.45           # minimum detection confidence
IOU_THRESHOLD   = 0.45           # NMS IOU threshold
MAX_AGE         = 30             # frames to keep a lost track alive
N_INIT          = 3              # frames before a track is 'confirmed'
NN_BUDGET       = 100            # max appearance feature history per track
FRAME_SKIP      = 1              # run YOLO every N frames (1 = every frame)
DISPLAY_FPS     = True
COLORS = {}                      # cache per-class colors


def get_color(class_id: int) -> tuple:
    """Generate a stable, visually distinct color per class ID."""
    if class_id not in COLORS:
        np.random.seed(class_id + 42)
        COLORS[class_id] = tuple(int(c) for c in np.random.randint(50, 230, 3))
    return COLORS[class_id]


def xyxy_to_xywh(box: np.ndarray) -> list:
    """Convert [x1, y1, x2, y2] → [x, y, w, h] (top-left origin).
    deep-sort-realtime expects xywh format."""
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def run_detection(model, frame: np.ndarray) -> list:
    """
    Run YOLO on a frame and return detections in Deep SORT format:
        [ ([x, y, w, h], confidence, class_id), ... ]
    """
    results = model(frame, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        boxes  = r.boxes.xyxy.cpu().numpy()   # shape (N, 4)
        confs  = r.boxes.conf.cpu().numpy()   # shape (N,)
        clss   = r.boxes.cls.cpu().numpy().astype(int)  # shape (N,)

        for box, conf, cls_id in zip(boxes, confs, clss):
            xywh = xyxy_to_xywh(box)
            detections.append((xywh, float(conf), int(cls_id)))
    return detections


def draw_track(frame: np.ndarray, track, class_names: dict) -> None:
    """Draw a single confirmed track's bounding box and label."""
    if not track.is_confirmed():
        return

    track_id = track.track_id
    ltrb = track.to_ltrb()          # [left, top, right, bottom]
    x1, y1, x2, y2 = (int(v) for v in ltrb)

    # Safely get class name — can be None on coasted tracks
    cls_id = track.get_det_class()
    label_name = class_names.get(cls_id, f"cls{cls_id}") if cls_id is not None else "?"
    color = get_color(cls_id if cls_id is not None else 0)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background + text
    label = f"{label_name} #{track_id}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label,
        (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )


def draw_hud(frame: np.ndarray, fps: float, n_tracks: int) -> None:
    """Overlay FPS and active track count in the top-left corner."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (210, 55), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"FPS : {fps:5.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Tracks: {n_tracks}", (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)


def main(source, save_output: bool):
    # ── Load model ────────────────────────────────────────────────────────
    print(f"[INFO] Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)
    class_names: dict = model.names   # {0: 'person', 1: 'bicycle', ...}

    # ── Init tracker ──────────────────────────────────────────────────────
    # embedder_gpu=False is safe for CPU; set True if you have CUDA
    print("[INFO] Initialising Deep SORT tracker...")
    tracker = DeepSort(
        max_age=MAX_AGE,
        n_init=N_INIT,
        nn_budget=NN_BUDGET,
        embedder="mobilenet",
        half=True,             # set False if GPU isn't available 
        bgr=True,               # OpenCV frames are BGR
        embedder_gpu=True,     # ← change to False if CUDA isn't available
    )

    # ── Open video source ─────────────────────────────────────────────────
    cap_source = 0 if source == "0" else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Source: {source}  |  {frame_w}x{frame_h} @ {src_fps:.1f} FPS")

    # ── Optional output writer ────────────────────────────────────────────
    writer = None
    if save_output:
        out_path = "tracked_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, src_fps, (frame_w, frame_h))
        print(f"[INFO] Saving output to: {out_path}")

    # ── Main loop ─────────────────────────────────────────────────────────
    frame_idx = 0
    fps_avg   = 0.0
    t_prev    = time.perf_counter()
    last_detections = []          # reuse detections on skipped frames

    print("[INFO] Running — press 'q' to quit, 's' to screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        # ── Detection (every FRAME_SKIP frames) ──────────────────────────
        if frame_idx % FRAME_SKIP == 0:
            last_detections = run_detection(model, frame)

        # ── Tracking (every frame) ────────────────────────────────────────
        # update_tracks needs the raw frame for appearance embedding
        tracked_objects = tracker.update_tracks(last_detections, frame=frame)

        # ── Draw ──────────────────────────────────────────────────────────
        confirmed = [t for t in tracked_objects if t.is_confirmed()]
        for track in confirmed:
            draw_track(frame, track, class_names)

        # ── FPS smoothing (exponential moving average) ────────────────────
        t_now   = time.perf_counter()
        instant = 1.0 / max(t_now - t_prev, 1e-6)
        fps_avg = 0.9 * fps_avg + 0.1 * instant
        t_prev  = t_now

        if DISPLAY_FPS:
            draw_hud(frame, fps_avg, len(confirmed))

        # ── Display / save ────────────────────────────────────────────────
        cv2.imshow("Detection & Tracking  [q=quit  s=screenshot]", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{frame_idx:05d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved: {fname}")

        frame_idx += 1

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_idx} frames at avg {fps_avg:.1f} FPS.")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 + Deep SORT real-time tracker")
    parser.add_argument(
        "--source", default="0",
        help="Video source: '0' for webcam, or path to a video file (default: 0)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated output to tracked_output.mp4"
    )
    args = parser.parse_args()
    main(args.source, args.save)

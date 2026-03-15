"""
real_time_yolov8_deepsort_plus.py

YOLOv8 (Ultralytics) + DeepSORT real-time tracking with:
- Stable track IDs
- Live per-class counts (current on-screen) + unique counts (ever seen)
- Alerts for selected classes (e.g., weapons)
- CSV logging (timestamp, track_id, class, confidence, bbox)
- Optional annotated video recording (MP4)
- Auto CPU/GPU selection (runs fine on CPU)

Install:
    pip install ultralytics opencv-python deep_sort_realtime numpy

Run examples:
    python real_time_yolov8_deepsort_plus.py --source 0 --save-out output.mp4
    python real_time_yolov8_deepsort_plus.py --source video5.mp4 --log detections_log.csv

Notes:
- Default model is yolov8n.pt (fast). You can switch to yolov8s.pt for a quality boost on CPU.
- If you don't have a GPU, leave --device as 'auto' (it will pick CPU).
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as e:
    raise RuntimeError(
        "deep_sort_realtime is required. Install with: pip install deep_sort_realtime"
    ) from e


# ----------------------------
# Utility helpers
# ----------------------------

def select_device_arg(user_device: str) -> str:
    """Return a device string acceptable to Ultralytics YOLO.
    - 'auto' -> 'cpu' if CUDA not available, else '0'
    - 'cpu' or CUDA strings are passed through
    """
    if user_device.lower() != 'auto':
        return user_device
    try:
        import torch
        return '0' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def class_color(class_id: int):
    # Simple deterministic color by class id
    rng = (int((class_id * 37) % 255), int((class_id * 17) % 255), int((class_id * 97) % 255))
    return rng


# ----------------------------
# Main app
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description='YOLOv8 + DeepSORT tracking with counts/alerts/logging/video save')
    ap.add_argument('--source', type=str, default='0', help='0 for webcam or path/URL to video/RTSP')
    ap.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path or name (e.g., yolov8n.pt)')
    ap.add_argument('--imgsz', type=int, default=640, help='inference image size')
    ap.add_argument('--conf', type=float, default=0.35, help='confidence threshold')
    ap.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    ap.add_argument('--device', type=str, default='auto', help="'auto', 'cpu', or CUDA string like '0' or '0,1'")

    # DeepSORT params (tuned for CPU)
    ap.add_argument('--max-age', type=int, default=30, help='DeepSORT max_age (frames)')
    ap.add_argument('--n-init', type=int, default=3, help='DeepSORT n_init (frames before confirming)')
    ap.add_argument('--max-iou-distance', type=float, default=0.7, help='DeepSORT max IoU distance')

    # Output & logging
    ap.add_argument('--save-out', type=str, default=None, help='path to save annotated video (e.g., output.mp4)')
    ap.add_argument('--save-fps', type=int, default=30, help='FPS for saved video')
    ap.add_argument('--log', type=str, default=None, help='CSV log path (e.g., detections_log.csv)')

    # Alerts
    ap.add_argument('--alert-classes', type=str, default='knife,scissors',
                    help='comma-separated class names that trigger on-screen alert')

    args = ap.parse_args()

    device = select_device_arg(args.device)

    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {args.model} on device={device}")
    model = YOLO(args.model)

    # Init DeepSORT
    print("Initializing DeepSORT tracker…")
    tracker = DeepSort(max_age=args.max_age,
                       max_iou_distance=args.max_iou_distance,
                       n_init=args.n_init)

    # Video capture
    src = args.source
    try:
        src = int(src)
    except Exception:
        pass
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    # Prepare output writer
    writer = None
    if args.save_out:
        out_path = Path(args.save_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare CSV logging
    csv_file = None
    csv_writer = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['timestamp', 'track_id', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])

    # Class names from model
    names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else {}

    # Alerts set
    alert_set = {n.strip().lower() for n in args.alert_classes.split(',') if n.strip()}

    # Counting structures
    unique_ids_by_class = defaultdict(set)   # class_name -> set(track_ids ever seen)

    # Timing for FPS
    prev_time = 0.0

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Lazy-create writer with actual frame size
        if writer is None and args.save_out:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(args.save_out), fourcc, float(args.save_fps), (w, h))

        # Run YOLOv8 inference
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device, verbose=False)
        result = results[0]

        # Build DeepSORT detections: ( [x,y,w,h], confidence, class )
        ds_inputs = []
        for det in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = det[:6]
            if score < args.conf:
                continue
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            ds_inputs.append(([x, y, w, h], float(score), int(cls)))

        tracks = tracker.update_tracks(ds_inputs, frame=frame)

        # Current counts per class (on screen now)
        current_counts = defaultdict(int)

        # Draw tracks
        alert_triggered = False
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            # Class id/name from track or last detection
            det_cls = None
            if hasattr(track, 'get_det_class'):
                det_cls = track.get_det_class()
            elif hasattr(track, 'det_class'):
                det_cls = track.det_class

            if isinstance(det_cls, (list, tuple)) and det_cls:
                det_cls = det_cls[-1]

            if det_cls is None:
                class_name = 'object'
                class_id = -1
            else:
                class_id = int(det_cls)
                class_name = names.get(class_id, str(class_id))

            # Confidence (best-effort)
            conf = None
            if hasattr(track, 'det_confidences') and track.det_confidences:
                try:
                    conf = float(track.det_confidences[-1])
                except Exception:
                    conf = None

            # Update live counts & unique counts
            current_counts[class_name] += 1
            unique_ids_by_class[class_name].add(track_id)

            # Draw rectangle & label
            color = class_color(class_id if class_id is not None else 0)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            label = f"ID {track_id} | {class_name}"
            if conf is not None:
                label += f" {conf:.2f}"
            cv2.putText(frame, label, (l, max(t - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Alerts
            if class_name.lower() in alert_set:
                alert_triggered = True

            # Log row
            if csv_writer is not None:
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                csv_writer.writerow([ts, track_id, class_name, f"{conf:.3f}" if conf is not None else '', l, t, r, b])

        # Draw counts panel
        y0 = 30
        cv2.putText(frame, 'Counts (current | unique):', (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y = y0 + 24
        for cls_name in sorted(set(list(current_counts.keys()) + list(unique_ids_by_class.keys()))):
            curr = current_counts.get(cls_name, 0)
            uniq = len(unique_ids_by_class.get(cls_name, set()))
            cv2.putText(frame, f"{cls_name}: {curr} | {uniq}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            y += 22

        # Show alert banner if triggered
        if alert_triggered:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
            cv2.putText(frame, 'ALERT! Monitored class detected', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0.0
        prev_time = now
        draw_fps(frame, fps)

        # Show & save
        cv2.imshow('YOLOv8 + DeepSORT', frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    if csv_writer is not None:
        csv_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

import argparse
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision.transforms import functional as TF
from ultralytics import YOLO

# MP3 playback via pygame (cross-platform).
try:
    import pygame
except ImportError:
    pygame = None


# -------------------- CONFIG --------------------
YOLO_PATH = r"gta3_yolov8m-3\weights\best.pt"
DEEPLAB_PATH = r"deeplabv3plus_resnet50_best.pth"

YOLO_IMGSZ = 1088
YOLO_CONF = 0.50       # YOLO confidence (separate from lane confidence)
LANE_CONFIDENCE = 0.86  # DeepLab road-class probability threshold
# Image-space vector distance thresholds (pixels).
# Euclidean distance between player bbox center and object bbox center.
# These are image-space pixels, NOT world-space meters.
VEHICLE_ALERT_VECTOR_PX = 300
PEDESTRIAN_ALERT_VECTOR_PX = 440
ROAD_CLASS = 1
PLAYER_CLASS = 0
VEHICLE_CLASS = 1
PEDESTRIAN_CLASS = 2

# Cooldown per alert, in seconds, to prevent audio spam.
ALERT_COOLDOWN_SEC = 2

# Replace these placeholder MP3 files with your own alert sounds.
AUDIO_FILES = {
    "lane": r"audio\lane_departure.mp3",
    "vehicle": r"audio\vehicle_alert.mp3",
    "pedestrian": r"audio\pedestrian_alert.mp3",
}

# DeepLab normalization from the existing project.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- MODELS --------------------
def load_models():
    yolo = YOLO(YOLO_PATH)

    deeplab = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=2,
        activation=None,
    ).to(DEVICE)

    checkpoint = torch.load(DEEPLAB_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    deeplab.load_state_dict(state_dict)
    deeplab.eval()

    print(f"Models loaded on {DEVICE}")
    print("YOLO names:", yolo.names)
    return yolo, deeplab


@torch.inference_mode()
def segment_road(frame, model):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(rgb)
    tensor = TF.normalize(tensor, MEAN, STD).unsqueeze(0).to(DEVICE)

    # Match the existing model's pad-to-multiple-of-16 preprocessing.
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

    with torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.float16,
        enabled=(DEVICE.type == "cuda"),
    ):
        logits = model(tensor)

    logits = logits[:, :, :h, :w]
    probabilities = torch.softmax(logits, dim=1)
    road_probability = probabilities[0, ROAD_CLASS].cpu().numpy()
    mask = (road_probability >= LANE_CONFIDENCE).astype(np.uint8)

    # Remove tiny isolated specks from the road mask.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


@torch.inference_mode()
def detect_objects(frame, model):
    result = model.predict(
        source=frame,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        device=0 if DEVICE.type == "cuda" else "cpu",
        verbose=False,
    )[0]

    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
        detections.append({
            "class_id": cls_id,
            "confidence": conf,
            "box": (int(x1), int(y1), int(x2), int(y2)),
        })
    return detections


# -------------------- LANE / ALERT LOGIC --------------------
def road_interval_at_row(mask, y, x_hint=None):
    """Return the contiguous road interval at y nearest x_hint, if any."""
    h, w = mask.shape
    y = int(np.clip(y, 0, h - 1))
    xs = np.flatnonzero(mask[y] > 0)
    if xs.size == 0:
        return None

    chunks = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    runs = [(int(c[0]), int(c[-1])) for c in chunks if c.size]
    if x_hint is None:
        return max(runs, key=lambda r: r[1] - r[0])

    def distance(run):
        left, right = run
        return 0 if left <= x_hint <= right else min(abs(x_hint-left), abs(x_hint-right))
    return min(runs, key=distance)


def point_on_road(mask, x, y, patch_radius=3, min_road_fraction=0.50):
    h, w = mask.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    x1, x2 = max(0, x-patch_radius), min(w, x+patch_radius+1)
    y1, y2 = max(0, y-patch_radius), min(h, y+patch_radius+1)
    patch = mask[y1:y2, x1:x2]
    return patch.size > 0 and float(patch.mean()) >= min_road_fraction


class PlaceholderAudio:
    """Plays MP3 alerts without blocking inference."""

    def __init__(self, cooldown=ALERT_COOLDOWN_SEC):
        self.cooldown = cooldown
        self.last_played = {}
        self.lock = threading.Lock()
        self.audio_paths = {}

        if pygame is None:
            print("[AUDIO] pygame missing. Run: pip install pygame")
            return

        try:
            pygame.mixer.init()
            print("[AUDIO] Mixer initialized:", pygame.mixer.get_init())
        except Exception as exc:
            print("[AUDIO] Mixer initialization failed:", exc)
            return

        for kind, audio_path in AUDIO_FILES.items():
            path = Path(audio_path).resolve()

            if path.exists():
                self.audio_paths[kind] = str(path)
                print(f"[AUDIO] Found {kind}: {path}")
            else:
                print(f"[AUDIO] FILE NOT FOUND: {path}")

    def alert(self, kind, message):
        now = time.monotonic()

        with self.lock:
            if now - self.last_played.get(kind, -1e9) < self.cooldown:
                return

            self.last_played[kind] = now

        print(f"[ALERT] {message}")

        audio_path = self.audio_paths.get(kind)
        if audio_path:
            try:
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                print(f"[AUDIO] Playing: {kind}")
            except Exception as exc:
                print(f"[AUDIO] Playback failed: {exc}")



def get_center(box):
    """Return the center point (x, y) of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def calculate_vector_distance(player_box, object_box):
    """
    Vector from player center to object center in image coordinates.
    Returns dx, dy and Euclidean magnitude. This is not physical distance.
    """
    player_cx, player_cy = get_center(player_box)
    object_cx, object_cy = get_center(object_box)
    dx = object_cx - player_cx
    dy = object_cy - player_cy
    magnitude = float(np.hypot(dx, dy))
    return dx, dy, magnitude


def is_object_ahead_in_lane(mask, box, player_box, frame_shape):
    """Check object is ahead of player and within the segmented road corridor."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = player_box
    obj_x = int((x1 + x2) / 2)
    obj_bottom = int(y2)
    player_x = int((px1 + px2) / 2)
    player_bottom = int(py2)

    if obj_bottom >= player_bottom - 8:
        return False, "not ahead"
    if not point_on_road(mask, obj_x, obj_bottom, patch_radius=5, min_road_fraction=0.40):
        return False, "outside road"

    lane_run = road_interval_at_row(mask, obj_bottom, x_hint=player_x)
    if lane_run is None:
        return False, "no road corridor"
    lane_left, lane_right = lane_run
    if not lane_left <= obj_x <= lane_right:
        return False, "outside driving corridor"
    return True, "ahead in driving corridor"

def draw_detection(frame, detection, label, color):
    x1, y1, x2, y2 = detection["box"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, f"{label} {detection['confidence']:.2f}",
        (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
    )


def process_frame(frame, yolo, deeplab, audio):
    mask = segment_road(frame, deeplab)
    detections = detect_objects(frame, yolo)

    overlay = frame.copy()
    overlay[mask > 0] = (
        0.55 * overlay[mask > 0] + 0.45 * np.array([40, 180, 40])
    ).astype(np.uint8)
    out = overlay

    player = next((d for d in detections if d["class_id"] == PLAYER_CLASS), None)
    vehicles = [d for d in detections if d["class_id"] == VEHICLE_CLASS]
    peds = [d for d in detections if d["class_id"] == PEDESTRIAN_CLASS]
    active_alerts = []

    if player is None:
        cv2.putText(out, "PLAYER NOT DETECTED", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        px1, py1, px2, py2 = player["box"]
        player_x = int((px1 + px2) / 2)
        player_contact_y = int(py2 - max(2, 0.08 * (py2 - py1)))
        on_road = point_on_road(mask, player_x, player_contact_y)
        draw_detection(out, player, "PLAYER", (255, 255, 0))

        if not on_road:
            audio.alert("lane", "LANE DEPARTURE")
            active_alerts.append("LANE DEPARTURE")
            status, status_color = "LANE DEPARTURE", (0, 0, 255)
        else:
            status, status_color = "ON ROAD", (0, 255, 0)
        cv2.putText(out, status, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        for d in vehicles:
            ahead_in_lane, reason = is_object_ahead_in_lane(
                mask, d["box"], player["box"], frame.shape
            )
            dx, dy, vector_distance = calculate_vector_distance(player["box"], d["box"])
            is_dangerous = ahead_in_lane and vector_distance <= VEHICLE_ALERT_VECTOR_PX

            label = f"VEHICLE d={vector_distance:.0f}px" if ahead_in_lane else "VEHICLE"
            color = (0, 0, 255) if is_dangerous else ((0, 165, 255) if ahead_in_lane else (255, 0, 0))
            draw_detection(out, d, label, color)

            if ahead_in_lane:
                p_center = tuple(map(int, get_center(player["box"])))
                o_center = tuple(map(int, get_center(d["box"])))
                cv2.arrowedLine(out, p_center, o_center, color, 2, tipLength=0.15)

            if is_dangerous:
                audio.alert("vehicle", f"VEHICLE AHEAD vector distance {vector_distance:.0f} pixels")
                active_alerts.append(f"VEHICLE CLOSE d={vector_distance:.0f}px")

        for d in peds:
            ahead_in_lane, reason = is_object_ahead_in_lane(
                mask, d["box"], player["box"], frame.shape
            )
            dx, dy, vector_distance = calculate_vector_distance(player["box"], d["box"])
            is_dangerous = ahead_in_lane and vector_distance <= PEDESTRIAN_ALERT_VECTOR_PX

            label = f"PEDESTRIAN d={vector_distance:.0f}px" if ahead_in_lane else "PEDESTRIAN"
            color = (0, 0, 255) if is_dangerous else ((255, 0, 255) if ahead_in_lane else (200, 0, 200))
            draw_detection(out, d, label, color)

            if ahead_in_lane:
                p_center = tuple(map(int, get_center(player["box"])))
                o_center = tuple(map(int, get_center(d["box"])))
                cv2.arrowedLine(out, p_center, o_center, color, 2, tipLength=0.15)

            if is_dangerous:
                audio.alert("pedestrian", f"PEDESTRIAN AHEAD vector distance {vector_distance:.0f} pixels")
                active_alerts.append(f"PEDESTRIAN CLOSE d={vector_distance:.0f}px")

    cv2.putText(out, f"Lane confidence: {LANE_CONFIDENCE:.2f}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(out, f"Vector alert: vehicle <= {VEHICLE_ALERT_VECTOR_PX:.0f}px | ped <= {PEDESTRIAN_ALERT_VECTOR_PX:.0f}px",
                (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    for i, label in enumerate(active_alerts[:3]):
        cv2.putText(out, label, (20, 120 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    for d in detections:
        if d["class_id"] not in (PLAYER_CLASS, VEHICLE_CLASS, PEDESTRIAN_CLASS):
            draw_detection(out, d, f"CLASS {d['class_id']}", (200, 200, 200))

    return out, mask


def main():
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", help="Input video path")
    source_group.add_argument("--live", action="store_true",
                              help="Capture live screen using MSS")
    parser.add_argument("--output", default="gta3_live_debug.mp4",
                        help="Annotated output video path")
    parser.add_argument("--monitor", type=int, default=1,
                        help="MSS monitor index (1 = primary monitor; 0 = all monitors)")
    parser.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Optional screen capture region: x y width height")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Output FPS for live capture")
    args = parser.parse_args()

    yolo, deeplab = load_models()
    audio = PlaceholderAudio()

    cap = None
    screen = None
    writer = None

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        title = "AI Driving Assistant - Video Test"
    else:
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError(
                "Live capture needs MSS. Install it with: pip install mss"
            ) from exc

        screen = mss.mss()
        monitors = screen.monitors
        if args.region:
            x, y, w, h = args.region
            capture_area = {"left": x, "top": y, "width": w, "height": h}
        else:
            if args.monitor < 0 or args.monitor >= len(monitors):
                raise ValueError(
                    f"Invalid monitor index {args.monitor}. Available: 0..{len(monitors)-1}"
                )
            capture_area = monitors[args.monitor]

        width = int(capture_area["width"])
        height = int(capture_area["height"])
        fps = max(1.0, args.fps)
        title = "AI Driving Assistant - Live Capture"
        print(f"Live capture area: {capture_area}")
        print("Switch to GTA III. Press Q in the assistant preview to stop.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    print(f"Saving annotated output to: {args.output}")

    try:
        while True:
            if args.video:
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                # MSS returns BGRA; OpenCV model pipeline expects BGR.
                raw = np.asarray(screen.grab(capture_area))
                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            annotated, _ = process_frame(frame, yolo, deeplab, audio)
            cv2.imshow(title, annotated)
            writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        if cap is not None:
            cap.release()
        if screen is not None:
            screen.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
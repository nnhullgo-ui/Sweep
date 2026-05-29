"""
batch_detect.py — Process folder of drone photos and write detections.json
in the format expected by the Sweep website (Lovable + Supabase).

Schema per entry (matches /admin upload expectations):
  {
    "lat": float,
    "lon": float,
    "label": "litter",
    "confidence": float (0-1),
    "timestamp": ISO string (e.g. "2026-05-29T14:23:45"),
    "photo_url": "" (empty unless you've pre-uploaded photos),
    "flight_id": "flight_2026-05-29_14-23"
  }

- For each photo: saves annotated copy + reads real GPS from EXIF.
- For each video: saves annotated copy + reads start-of-recording GPS +
  extracts top 5 detection frames.
- Writes detections.json with one entry PER detection (not per photo).

Run with: python3 batch_detect.py
"""
from pathlib import Path
from datetime import datetime
import json
import subprocess
import cv2
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from ultralytics import YOLO

# --- Configuration ---
HERE = Path(__file__).parent.resolve()
INPUT_FOLDER = Path.home() / "Desktop" / "Photos"
OUTPUT_FOLDER = Path.home() / "Desktop" / "Sweep_Results"
MODEL_PATH = HERE / "models" / "best.pt"
DASHBOARD_JSON = HERE / "detections.json"
CONFIDENCE = 0.6
TOP_FRAMES_PER_VIDEO = 5

# Auto-generated flight ID tags every detection from this run.
# Format: flight_YYYY-MM-DD_HH-MM (minute-level granularity)
FLIGHT_ID = f"flight_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}


def read_gps_from_photo(image_path):
    """Extract lat/lon from a photo's EXIF. Returns (lat, lon) or (None, None)."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None, None
        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value
                break
        if not gps_info:
            return None, None

        def dms_to_decimal(dms, ref):
            decimal = float(dms[0]) + (float(dms[1]) / 60) + (float(dms[2]) / 3600)
            if ref in ("S", "W"):
                decimal = -decimal
            return decimal

        lat = dms_to_decimal(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
        lon = dms_to_decimal(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])
        return lat, lon
    except Exception as e:
        print(f"   ! Could not read photo GPS: {e}")
        return None, None


def read_gps_from_video(video_path):
    """Extract start-of-recording GPS from MP4 metadata using exiftool.
    Returns (lat, lon) or (None, None)."""
    try:
        result = subprocess.run(
            ["exiftool", "-GPSLatitude", "-GPSLongitude",
             "-GPSLatitudeRef", "-GPSLongitudeRef",
             "-c", "%+.8f", str(video_path)],
            capture_output=True, text=True, check=True
        )
        lat = lon = None
        lat_ref = lon_ref = None
        for line in result.stdout.splitlines():
            if ":" not in line:
                continue
            key, value = [x.strip() for x in line.split(":", 1)]
            if key == "GPS Latitude":
                lat = float(value.replace("+", ""))
            elif key == "GPS Longitude":
                lon = float(value.replace("+", ""))
            elif key == "GPS Latitude Ref":
                lat_ref = value
            elif key == "GPS Longitude Ref":
                lon_ref = value
        if lat is not None and lat_ref and lat_ref.lower().startswith("s"):
            lat = -lat
        if lon is not None and lon_ref and lon_ref.lower().startswith("w"):
            lon = -lon
        return lat, lon
    except Exception as e:
        print(f"   ! Could not read video GPS: {e}")
        return None, None


def process_photo(photo_path, model, entries):
    """Run detection on a photo, save annotated copy, append detections to list."""
    print(f"\n[PHOTO] {photo_path.name}")
    lat, lon = read_gps_from_photo(photo_path)
    if lat is None or lon is None:
        print("   ! No GPS in EXIF, skipping (cannot map without coordinates).")
        return
    print(f"   GPS: {lat:.6f}, {lon:.6f}")

    results = model.predict(source=str(photo_path), conf=CONFIDENCE,
                            save=False, verbose=False)
    r = results[0]
    n_dets = len(r.boxes) if r.boxes is not None else 0
    print(f"   Detections: {n_dets}")

    # Save annotated copy
    annotated = r.plot()
    out_path = OUTPUT_FOLDER / f"annotated_{photo_path.name}"
    cv2.imwrite(str(out_path), annotated)

    if n_dets == 0:
        return

    # Use file modification time as the timestamp (closer to when the photo
    # was taken than "now"). Falls back to now() if anything fails.
    try:
        mtime = datetime.fromtimestamp(photo_path.stat().st_mtime)
        timestamp = mtime.isoformat(timespec="seconds")
    except Exception:
        timestamp = datetime.now().isoformat(timespec="seconds")

    names = r.names
    for box in r.boxes:
        cls_id = int(box.cls.item())
        entries.append({
            "lat": lat,
            "lon": lon,
            "label": names.get(cls_id, "litter"),
            "confidence": round(float(box.conf.item()), 3),
            "timestamp": timestamp,
            "photo_url": "",          # fill in later via /admin upload
            "flight_id": FLIGHT_ID,
            "source_image": photo_path.name,
        })


def process_video(video_path, model, entries):
    """Run detection on a video, save annotated copy + top frames.
    Top-frame detections are added to entries with the video's start GPS."""
    print(f"\n[VIDEO] {video_path.name}")
    lat, lon = read_gps_from_video(video_path)
    if lat is None or lon is None:
        print("   ! No GPS in video metadata, skipping.")
        return
    print(f"   GPS (start-of-recording): {lat:.6f}, {lon:.6f}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("   ! Could not open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Score each frame by total detection count, keep the top N
    frame_scores = []  # list of (frame_index, n_dets, max_conf, boxes_data)
    SAMPLE_STRIDE = max(1, int(fps))  # sample 1 frame per second

    print(f"   Scanning {total} frames @ {fps:.1f}fps (stride {SAMPLE_STRIDE})...")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % SAMPLE_STRIDE == 0:
            results = model.predict(source=frame, conf=CONFIDENCE,
                                    save=False, verbose=False)
            r = results[0]
            n_dets = len(r.boxes) if r.boxes is not None else 0
            if n_dets > 0:
                box_data = [
                    {"cls": int(b.cls.item()), "conf": float(b.conf.item())}
                    for b in r.boxes
                ]
                max_conf = max(b["conf"] for b in box_data)
                frame_scores.append((idx, n_dets, max_conf, box_data, frame.copy()))
        idx += 1
    cap.release()

    if not frame_scores:
        print("   No detections in any sampled frame.")
        return

    # Sort by detection count desc, then by max confidence desc
    frame_scores.sort(key=lambda x: (-x[1], -x[2]))
    top = frame_scores[:TOP_FRAMES_PER_VIDEO]
    print(f"   Saving top {len(top)} frames with detections.")

    base_time = datetime.fromtimestamp(video_path.stat().st_mtime)
    base_iso = base_time.isoformat(timespec="seconds")

    for frame_idx, n_dets, max_conf, box_data, frame_img in top:
        # Save annotated frame
        results = model.predict(source=frame_img, conf=CONFIDENCE,
                                save=False, verbose=False)
        annotated = results[0].plot()
        out_name = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(OUTPUT_FOLDER / out_name), annotated)

        # Append one entry per detection in this frame
        for box in box_data:
            entries.append({
                "lat": lat,
                "lon": lon,
                "label": "litter",
                "confidence": round(box["conf"], 3),
                "timestamp": base_iso,
                "photo_url": "",
                "flight_id": FLIGHT_ID,
                "source_image": out_name,
            })


def main():
    if not INPUT_FOLDER.exists():
        raise SystemExit(f"Input folder not found: {INPUT_FOLDER}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print(f"Flight ID: {FLIGHT_ID}")
    print(f"Confidence threshold: {CONFIDENCE}")
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    photos = sorted([p for p in INPUT_FOLDER.iterdir() if p.suffix in PHOTO_EXTS])
    videos = sorted([p for p in INPUT_FOLDER.iterdir() if p.suffix in VIDEO_EXTS])
    print(f"Found {len(photos)} photo(s) and {len(videos)} video(s).")

    entries = []

    for p in photos:
        process_photo(p, model, entries)
    for v in videos:
        process_video(v, model, entries)

    DASHBOARD_JSON.write_text(json.dumps(entries, indent=2))

    print()
    print("=" * 60)
    print(f"Wrote {len(entries)} detection entries to: {DASHBOARD_JSON}")
    print(f"Annotated files in: {OUTPUT_FOLDER}")
    print(f"Flight ID for this batch: {FLIGHT_ID}")
    print()
    print("Next step: upload detections.json at the Sweep site /admin page.")


if __name__ == "__main__":
    main()
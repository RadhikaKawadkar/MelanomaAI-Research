"""
convert_tfrecords.py
Converts ISIC melanoma .tfrec files → JPG images
Output: dataset/benign/ and dataset/malignant/
Usage:  python convert_tfrecords.py
"""

import os
import glob
import csv
import io
import sys
import logging
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────
TFREC_DIR   = os.path.join("data", "v2", "raw", "tfrecords")
CSV_PATH    = os.path.join("data", "v2", "raw", "train.csv")
OUTPUT_DIR  = os.path.join("data", "v2", "processed")
IMAGE_SIZE  = 384                  # preserve 384×384
BENIGN_DIR  = os.path.join(OUTPUT_DIR, "benign")
MALIGNANT_DIR = os.path.join(OUTPUT_DIR, "malignant")
LOG_FILE    = os.path.join("data", "v2", "raw", "convert_log.txt")
# ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def load_label_map(csv_path: str) -> dict:
    """Load image_name → target (0=benign, 1=malignant) from CSV."""
    label_map = {}
    if not os.path.exists(csv_path):
        log.warning(f"CSV not found: {csv_path} — will use TFRecord labels if available")
        return label_map
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("image_name") or row.get("image") or ""
            target = row.get("target") or row.get("label") or "0"
            name = name.strip().replace(".jpg", "").replace(".jpeg", "")
            label_map[name] = int(target.strip())
    log.info(f"Loaded {len(label_map)} labels from {csv_path}")
    return label_map


def parse_tfrecord(record_bytes):
    """Parse a single TFRecord — handles multiple common feature schemas."""
    feature_desc = {
        "image":        tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image/encoded":tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image_name":   tf.io.FixedLenFeature([], tf.string, default_value=""),
        "target":       tf.io.FixedLenFeature([], tf.int64,  default_value=-1),
    }
    try:
        parsed = tf.io.parse_single_example(record_bytes, feature_desc)
        # image bytes — try both keys
        img_bytes = parsed["image"] if parsed["image"].numpy() else parsed["image/encoded"].numpy()
        if not img_bytes:
            return None, None, None
        image_name = parsed["image_name"].numpy().decode("utf-8").strip()
        target     = int(parsed["target"].numpy())
        return img_bytes, image_name, target
    except Exception:
        return None, None, None


def decode_and_save(img_bytes, out_path: str) -> bool:
    """Decode raw bytes → JPG and save to out_path."""
    try:
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        img = tf.cast(img, tf.uint8)
        jpg = tf.image.encode_jpeg(img, quality=95)
        with open(out_path, "wb") as f:
            f.write(jpg.numpy())
        return True
    except Exception:
        return False


def convert(tfrec_files: list, label_map: dict):
    """Main conversion loop."""
    os.makedirs(BENIGN_DIR,    exist_ok=True)
    os.makedirs(MALIGNANT_DIR, exist_ok=True)

    total = saved = skipped = corrupted = 0
    label_source = "csv" if label_map else "tfrec"

    log.info(f"Found {len(tfrec_files)} .tfrec file(s) — label source: {label_source}")

    for tfrec_path in tfrec_files:
        log.info(f"Processing: {tfrec_path}")
        dataset = tf.data.TFRecordDataset(tfrec_path, compression_type="")

        for record in tqdm(dataset, desc=os.path.basename(tfrec_path), unit="img"):
            total += 1
            try:
                img_bytes, image_name, tfrec_target = parse_tfrecord(record)

                if img_bytes is None:
                    corrupted += 1
                    continue

                # resolve label: CSV takes priority
                if label_map and image_name in label_map:
                    target = label_map[image_name]
                elif tfrec_target in (0, 1):
                    target = tfrec_target
                else:
                    # fallback: try stripping extension variations
                    clean = image_name.replace(".jpg","").replace(".jpeg","")
                    if clean in label_map:
                        target = label_map[clean]
                    else:
                        log.debug(f"No label for {image_name} — skipping")
                        skipped += 1
                        continue

                out_folder = MALIGNANT_DIR if target == 1 else BENIGN_DIR
                fname = f"{image_name}.jpg" if not image_name.endswith(".jpg") else image_name
                out_path = os.path.join(out_folder, fname)

                if os.path.exists(out_path):
                    skipped += 1
                    continue

                ok = decode_and_save(img_bytes, out_path)
                if ok:
                    saved += 1
                else:
                    corrupted += 1

            except Exception as e:
                corrupted += 1
                log.debug(f"Record error: {e}")
                continue

    return total, saved, skipped, corrupted


def main():
    log.info("=" * 60)
    log.info("MelanomaAI — TFRecords → JPG Converter")
    log.info("=" * 60)

    tfrec_files = sorted(glob.glob(os.path.join(TFREC_DIR, "**", "*.tfrec"), recursive=True))
    if not tfrec_files:
        tfrec_files = sorted(glob.glob(os.path.join(TFREC_DIR, "**", "*.tfrecord"), recursive=True))

    if not tfrec_files:
        log.error("No .tfrec or .tfrecord files found. Check TFREC_DIR path.")
        sys.exit(1)

    label_map = load_label_map(CSV_PATH)
    total, saved, skipped, corrupted = convert(tfrec_files, label_map)

    # ── Summary ──
    benign_count    = len(glob.glob(os.path.join(BENIGN_DIR,    "*.jpg")))
    malignant_count = len(glob.glob(os.path.join(MALIGNANT_DIR, "*.jpg")))

    log.info("=" * 60)
    log.info("CONVERSION COMPLETE")
    log.info(f"  Total records   : {total}")
    log.info(f"  Saved           : {saved}")
    log.info(f"  Skipped (exist) : {skipped}")
    log.info(f"  Corrupted       : {corrupted}")
    log.info(f"  Benign images   : {benign_count}")
    log.info(f"  Malignant images: {malignant_count}")
    ratio = benign_count / malignant_count if malignant_count > 0 else 0
    log.info(f"  Class ratio     : {ratio:.1f}:1 (benign:malignant)")
    log.info(f"  Output folder   : {os.path.abspath(OUTPUT_DIR)}")
    log.info(f"  Log saved       : {LOG_FILE}")
    log.info("=" * 60)

    if malignant_count == 0:
        log.warning("WARNING: 0 malignant images saved — check CSV labels or TFRecord schema")


if __name__ == "__main__":
    main()
"""
balance_dataset.py — MelanomaAI v2
Balances skin cancer dataset for training.

Pipeline:
  1. Validate all images in source dirs
  2. Undersample benign  → TARGET_BENIGN
  3. Augment malignant   → TARGET_MALIGNANT
  4. Write balanced set  → data/v2/balanced/

Usage:
  python balance_dataset.py

Output:
  data/v2/balanced/
      benign/
      malignant/
"""

import os
import sys
import random
import shutil
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SRC_BENIGN      = "data/v2/processed/benign"
SRC_MALIGNANT   = "data/v2/processed/malignant"

DST_ROOT        = "data/v2/balanced"
DST_BENIGN      = os.path.join(DST_ROOT, "benign")
DST_MALIGNANT   = os.path.join(DST_ROOT, "malignant")

TARGET_BENIGN    = 15_000
TARGET_MALIGNANT =  5_000

SEED             = 42
IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".bmp"}
LOG_FILE         = "balance_log.txt"
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─── AUGMENTATION PIPELINE ────────────────────────────────────────────────────
AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05,
                       alpha_affine=120 * 0.03, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10,
                         sat_shift_limit=20,
                         val_shift_limit=10, p=0.4),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
])


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def list_images(folder: str) -> list[Path]:
    """Return sorted list of valid image paths in folder."""
    folder = Path(folder)
    if not folder.exists():
        log.error(f"Source folder not found: {folder}")
        sys.exit(1)
    paths = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )
    return paths


def is_valid_image(path: Path) -> bool:
    """Return True if image can be decoded successfully."""
    try:
        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except Exception:
        return False


def validate_images(paths: list[Path], label: str) -> list[Path]:
    """Filter out corrupted images with progress bar."""
    valid = []
    log.info(f"Validating {len(paths)} {label} images...")
    for p in tqdm(paths, desc=f"Validate {label}", unit="img"):
        if is_valid_image(p):
            valid.append(p)
        else:
            log.warning(f"Corrupt/unreadable — skipped: {p.name}")
    log.info(f"  Valid {label}: {len(valid)} / {len(paths)}")
    return valid


def safe_copy(src: Path, dst_dir: str, prefix: str = "") -> bool:
    """Copy image to dst_dir with optional prefix; skip if exists."""
    fname = f"{prefix}{src.name}" if prefix else src.name
    dst   = Path(dst_dir) / fname
    counter = 1
    while dst.exists():
        stem   = src.stem
        suffix = src.suffix
        dst    = Path(dst_dir) / f"{prefix}{stem}_{counter}{suffix}"
        counter += 1
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        log.warning(f"Copy failed {src.name}: {e}")
        return False


def augment_and_save(src: Path, dst_dir: str, aug_id: int) -> bool:
    """Apply augmentation pipeline and save result."""
    try:
        img = cv2.imread(str(src))
        if img is None:
            return False
        img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = AUG(image=img_rgb)["image"]
        out_img   = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        stem   = src.stem
        suffix = src.suffix if src.suffix.lower() in {".jpg", ".jpeg"} else ".jpg"
        dst    = Path(dst_dir) / f"aug_{aug_id:06d}_{stem}{suffix}"
        counter = 1
        while dst.exists():
            dst = Path(dst_dir) / f"aug_{aug_id:06d}_{stem}_{counter}{suffix}"
            counter += 1

        cv2.imwrite(str(dst), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        log.debug(f"Augmentation failed {src.name}: {e}")
        return False


def count_images(folder: str) -> int:
    return len(list(Path(folder).glob("*.jpg"))) + \
           len(list(Path(folder).glob("*.jpeg"))) + \
           len(list(Path(folder).glob("*.png")))


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("MelanomaAI v2 — Dataset Balancer")
    log.info("=" * 65)

    os.makedirs(DST_BENIGN,    exist_ok=True)
    os.makedirs(DST_MALIGNANT, exist_ok=True)

    src_benign_all    = list_images(SRC_BENIGN)
    src_malignant_all = list_images(SRC_MALIGNANT)
    log.info(f"\nBEFORE BALANCING:")
    log.info(f"  Benign    source : {len(src_benign_all)}")
    log.info(f"  Malignant source : {len(src_malignant_all)}")
    ratio_before = len(src_benign_all) / max(len(src_malignant_all), 1)
    log.info(f"  Ratio            : {ratio_before:.1f}:1\n")

    benign_valid    = validate_images(src_benign_all,    "benign")
    malignant_valid = validate_images(src_malignant_all, "malignant")

    log.info(f"\n[BENIGN] Undersampling {len(benign_valid)} → {TARGET_BENIGN}")
    random.shuffle(benign_valid)
    benign_selected = benign_valid[:TARGET_BENIGN]

    benign_copied = 0
    for img_path in tqdm(benign_selected, desc="Copy benign", unit="img"):
        if safe_copy(img_path, DST_BENIGN):
            benign_copied += 1

    log.info(f"  Benign copied: {benign_copied}")

    log.info(f"\n[MALIGNANT] Building {TARGET_MALIGNANT} from {len(malignant_valid)} originals")

    orig_copied = 0
    for img_path in tqdm(malignant_valid, desc="Copy malignant originals", unit="img"):
        if safe_copy(img_path, DST_MALIGNANT, prefix="orig_"):
            orig_copied += 1

    log.info(f"  Originals copied: {orig_copied}")

    current_malignant = count_images(DST_MALIGNANT)
    needed            = max(0, TARGET_MALIGNANT - current_malignant)
    log.info(f"  Need {needed} augmented images to reach {TARGET_MALIGNANT}")

    aug_saved  = 0
    aug_failed = 0
    aug_id     = 0

    if needed > 0:
        pool = malignant_valid.copy()
        random.shuffle(pool)
        pool_cycle = pool * (needed // len(pool) + 2)

        pbar = tqdm(total=needed, desc="Augment malignant", unit="img")
        for img_path in pool_cycle:
            if aug_saved >= needed:
                break
            ok = augment_and_save(img_path, DST_MALIGNANT, aug_id)
            aug_id += 1
            if ok:
                aug_saved += 1
                pbar.update(1)
            else:
                aug_failed += 1
        pbar.close()

    final_benign    = count_images(DST_BENIGN)
    final_malignant = count_images(DST_MALIGNANT)
    ratio_after     = final_benign / max(final_malignant, 1)

    log.info("\n" + "=" * 65)
    log.info("BALANCING COMPLETE")
    log.info(f"  Source benign          : {len(src_benign_all)}")
    log.info(f"  Source malignant       : {len(src_malignant_all)}")
    log.info(f"  ─────────────────────────────")
    log.info(f"  Final benign           : {final_benign}")
    log.info(f"  Final malignant        : {final_malignant}")
    log.info(f"    └─ originals copied  : {orig_copied}")
    log.info(f"    └─ augmented added   : {aug_saved}")
    log.info(f"    └─ augment failures  : {aug_failed}")
    log.info(f"  Final ratio            : {ratio_after:.1f}:1 (benign:malignant)")
    log.info(f"  Output                 : {os.path.abspath(DST_ROOT)}")
    log.info(f"  Log                    : {LOG_FILE}")
    log.info("=" * 65)

    if final_malignant < TARGET_MALIGNANT * 0.9:
        log.warning("WARNING: Malignant count below 90% of target — check source images")
    if ratio_after > 4.0:
        log.warning(f"WARNING: Ratio {ratio_after:.1f}:1 still high — consider increasing TARGET_MALIGNANT")
    else:
        log.info("✅ Dataset ready for training with torchvision ImageFolder")


if __name__ == "__main__":
    main()

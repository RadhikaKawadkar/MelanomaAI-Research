import os
import shutil
import pandas as pd

# Paths
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

metadata = pd.read_csv(os.path.join(RAW_PATH, "HAM10000_metadata.csv"))

# Keep only mel and nv
metadata = metadata[metadata["dx"].isin(["mel", "nv"])]

# Create class folders
benign_path = os.path.join(PROCESSED_PATH, "benign")
malignant_path = os.path.join(PROCESSED_PATH, "malignant")

os.makedirs(benign_path, exist_ok=True)
os.makedirs(malignant_path, exist_ok=True)

# Image folders
img_folder1 = os.path.join(RAW_PATH, "HAM10000_images_part_1")
img_folder2 = os.path.join(RAW_PATH, "HAM10000_images_part_2")

for _, row in metadata.iterrows():
    image_id = row["image_id"] + ".jpg"
    label = row["dx"]

    if os.path.exists(os.path.join(img_folder1, image_id)):
        source = os.path.join(img_folder1, image_id)
    else:
        source = os.path.join(img_folder2, image_id)

    if label == "nv":
        destination = os.path.join(benign_path, image_id)
    else:
        destination = os.path.join(malignant_path, image_id)

    shutil.copyfile(source, destination)

print("Dataset prepared successfully!")
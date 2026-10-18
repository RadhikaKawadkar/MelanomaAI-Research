import unittest
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from split_dataset import partition_dataset_grouped, filter_leak_free_images


class TestSplitDataset(unittest.TestCase):
    def setUp(self):
        # Create a synthetic metadata DataFrame simulating HAM10000 with multiple images per lesion
        data = {
            "lesion_id": [
                "HAM_001", "HAM_001", "HAM_002", "HAM_003", "HAM_004", "HAM_004",
                "HAM_005", "HAM_006", "HAM_007", "HAM_008", "HAM_008", "HAM_009",
                "HAM_010", "HAM_011", "HAM_012", "HAM_013", "HAM_014", "HAM_015"
            ],
            "image_id": [f"ISIC_{i:07d}" for i in range(18)],
            "dx": [
                "mel", "mel", "nv", "nv", "mel", "mel",
                "nv", "nv", "nv", "mel", "mel", "nv",
                "nv", "nv", "mel", "nv", "nv", "nv"
            ],
        }
        self.df = pd.DataFrame(data)

    def test_grouped_partition_zero_overlap(self):
        splits = partition_dataset_grouped(
            self.df,
            group_col="lesion_id",
            label_col="dx",
            positive_label="mel",
            train_ratio=0.6,
            calib_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        train_lesions = set(splits["train"]["lesion_id"])
        calib_lesions = set(splits["calibration"]["lesion_id"])
        test_lesions = set(splits["test"]["lesion_id"])

        # Crucial assertions: Absolutely zero lesion overlap between splits
        self.assertEqual(len(train_lesions.intersection(calib_lesions)), 0)
        self.assertEqual(len(train_lesions.intersection(test_lesions)), 0)
        self.assertEqual(len(calib_lesions.intersection(test_lesions)), 0)

        # Check total image coverage
        total_images = len(splits["train"]) + len(splits["calibration"]) + len(splits["test"])
        self.assertEqual(total_images, len(self.df))

    def test_filter_leak_free_images(self):
        test_paths = [
            Path("data/malignant/ISIC_001.jpg"),
            Path("data/malignant/aug_00001_ISIC_001.jpg"),
            Path("data/malignant/aug_00002_ISIC_002.jpg"),
            Path("data/benign/ISIC_003.jpg"),
        ]
        clean = filter_leak_free_images(test_paths, is_test_or_val=True)
        # Should only retain genuine original clinical images
        self.assertEqual(len(clean), 2)
        self.assertTrue(all(not p.name.startswith("aug_") for p in clean))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
from collections import defaultdict

import hashlib
import numpy as np
import pandas as pd

from PIL import Image

from src.logger import get_logger
from src.exception import (
    DatasetNotFoundException,
    DataValidationException
)

from src.data.data_validation.validate_csv import validate_csv

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_reports"
)

NEAR_DUPLICATE_THRESHOLD = 5


def calculate_image_hash(image_path):

    try:
        with Image.open(image_path) as image:

            image = image.convert("RGB")

            image_hash = hashlib.sha256()

            image_hash.update(str(image.size).encode())

            image_hash.update(image.tobytes())

            return image_hash.hexdigest()

    except Exception as error:

        logger.exception(
            f"Failed to calculate image hash: {image_path}"
        )

        raise DataValidationException(
            f"Image hashing failed: {image_path}"
        ) from error


def detect_exact_duplicates(df):

    hash_groups = defaultdict(list)

    try:
        for _, row in df.iterrows():

            image_path = PROJECT_ROOT / row["image_path"]

            image_hash = calculate_image_hash(image_path)

            hash_groups[image_hash].append({
                "image_path": row["image_path"],
                "label": row["label"],
                "split": row["split"]
            })

        duplicates = []

        for image_hash, records in hash_groups.items():

            if len(records) > 1:

                for record in records:
                    duplicates.append({
                        "hash": image_hash,
                        "image_path": record["image_path"],
                        "label": record["label"],
                        "split": record["split"],
                        "duplicate_count": len(records)
                    })

        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        report_file = REPORT_DIR / "exact_duplicates.csv"

        duplicate_df = pd.DataFrame(duplicates)

        duplicate_df.to_csv(report_file, index=False)

        if duplicates:

            logger.warning(
                f"Exact duplicate images found: {len(duplicates)} records"
            )

        else:

            logger.info("No exact duplicate images found.")

        logger.info(f"Duplicate report saved: {report_file}")

        return duplicate_df

    except Exception as error:

        logger.exception("Exact duplicate detection failed.")

        raise DataValidationException(
            "Exact duplicate detection failed."
        ) from error


def detect_split_leakage(duplicate_df):

    try:
        leakage_records = []

        if duplicate_df.empty:
            logger.info("No exact duplicates available for leakage analysis.")
            return pd.DataFrame()

        grouped = duplicate_df.groupby("hash")

        for image_hash, group in grouped:

            splits = group["split"].unique()

            if len(splits) > 1:

                leakage_records.append({
                    "hash": image_hash,
                    "image_paths": " | ".join(
                        group["image_path"].tolist()
                    ),
                    "labels": " | ".join(
                        group["label"].astype(str).tolist()
                    ),
                    "splits": " | ".join(splits),
                    "record_count": len(group)
                })

        leakage_df = pd.DataFrame(leakage_records)

        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        report_file = REPORT_DIR / "split_leakage.csv"

        leakage_df.to_csv(report_file, index=False)

        if not leakage_df.empty:

            logger.warning(
                f"Train-validation leakage groups found: "
                f"{len(leakage_df)}"
            )

        else:

            logger.info(
                "No exact duplicate leakage between splits found."
            )

        logger.info(f"Leakage report saved: {report_file}")

        return leakage_df

    except Exception as error:

        logger.exception("Split leakage detection failed.")

        raise DataValidationException(
            "Split leakage detection failed."
        ) from error



if __name__ == "__main__":

    df = validate_csv()

    duplicates = detect_exact_duplicates(df)

    leakage = detect_split_leakage(duplicates)

    print("\nExact duplicate records:", len(duplicates))
    print("Train-validation leakage groups:", len(leakage))

    print("\nFirst 10 leakage groups:")
    print(leakage.head(10).to_string(index=False))
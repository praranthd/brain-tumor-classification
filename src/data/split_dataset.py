
import shutil
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.logger import get_logger
from src.exception import DataValidationException

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_DIR = PROJECT_ROOT / "data" / "raw" / "Training"


OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "splits.csv"
BACKUP_FILE = PROJECT_ROOT / "data" / "processed" / "splits_backup.csv"


CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# --------------------------------------------------
# Backup existing split
# --------------------------------------------------

def backup_existing_split():

    if OUTPUT_FILE.exists():

        if BACKUP_FILE.exists():
            logger.warning("Backup file already exists. It will not be overwritten.")
            return

        shutil.copy2(OUTPUT_FILE, BACKUP_FILE)
        logger.info(f"Previous split backed up: {BACKUP_FILE}")

    else:
        logger.info("No existing split file found. Backup skipped.")


# --------------------------------------------------
# Calculate exact image hash
# --------------------------------------------------

def calculate_image_hash(image_path):

    try:

        from PIL import Image

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            image_hash = hashlib.sha256()
            image_hash.update(str(image.size).encode())
            image_hash.update(image.tobytes())

            return image_hash.hexdigest()

    except Exception as error:

        logger.exception(f"Image hashing failed: {image_path}")

        raise DataValidationException(
            f"Failed to hash image: {image_path}"
        ) from error



# --------------------------------------------------
# Collect images
# --------------------------------------------------

def collect_image_records():

    records = []

    for class_name in CLASSES:

        class_dir = TRAINING_DIR / class_name

        if not class_dir.exists():

            raise DataValidationException(
                f"Class folder not found: {class_dir}"
            )

        for image_path in class_dir.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower() in VALID_EXTENSIONS
            ):

                records.append({
                    "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                    "label": class_name
                })

    df = pd.DataFrame(records)

    if df.empty:

        raise DataValidationException(
            "No images found in training dataset."
        )

    logger.info(f"Total images collected: {len(df)}")

    return df


# --------------------------------------------------
# Generate group-aware split
# --------------------------------------------------

def create_group_aware_split(df):

    try:

        logger.info("Calculating image hashes.")

        df["group_id"] = df["image_path"].apply(
            lambda path: calculate_image_hash(
                PROJECT_ROOT / path
            )
        )

        logger.info("Image hashing completed.")

        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        train_indices, val_indices = next(
            splitter.split(
                df,
                y=df["label"],
                groups=df["group_id"]
            )
        )

        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()

        train_df["split"] = "train"
        val_df["split"] = "validation"

        split_df = pd.concat(
            [train_df, val_df],
            ignore_index=True
        )

        # Remove hash column from final manifest
        split_df = split_df.drop(columns=["group_id"])

        return split_df, train_df, val_df

    except Exception as error:

        logger.exception("Group-aware splitting failed.")

        raise DataValidationException(
            "Failed to generate group-aware split."
        ) from error


# --------------------------------------------------
# Save split
# --------------------------------------------------

def save_split(split_df):

    try:

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        split_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        logger.info(f"Split manifest saved: {OUTPUT_FILE}")

    except Exception as error:

        logger.exception("Failed to save split manifest.")

        raise DataValidationException(
            "Failed to save split CSV."
        ) from error


# --------------------------------------------------
# Report
# --------------------------------------------------

def report_split(train_df, val_df, split_df):

    logger.info(f"Total images: {len(split_df)}")
    logger.info(f"Training images: {len(train_df)}")
    logger.info(f"Validation images: {len(val_df)}")

    distribution = pd.crosstab(
        split_df["split"],
        split_df["label"]
    )

    logger.info(f"\nClass distribution:\n{distribution}")


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

def main():

    logger.info("Starting group-aware dataset splitting.")

    backup_existing_split()

    df = collect_image_records()

    split_df, train_df, val_df = create_group_aware_split(df)

    save_split(split_df)

    report_split(
        train_df,
        val_df,
        split_df
    )

    logger.info("Dataset splitting completed successfully.")


if __name__ == "__main__":

    main()

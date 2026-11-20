
from pathlib import Path

import pandas as pd
from PIL import Image

from src.logger import get_logger
from src.exception import (
    DatasetNotFoundException,
    DataValidationException
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANIFEST_FILE = (
    PROJECT_ROOT / "data" / "processed" / "test_manifest.csv"
)

OVERLAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_reports"
    / "train_test_overlap.csv"
)

REQUIRED_COLUMNS = {"image_path", "label"}

VALID_LABELS = {
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_manifest():

    logger.info("Loading clean test manifest.")

    if not MANIFEST_FILE.exists():

        raise DatasetNotFoundException(
            f"Test manifest not found: {MANIFEST_FILE}"
        )

    df = pd.read_csv(MANIFEST_FILE)

    if df.empty:

        raise DataValidationException(
            "Test manifest is empty."
        )

    if not REQUIRED_COLUMNS.issubset(df.columns):

        raise DataValidationException(
            "Required columns are missing from test manifest."
        )

    logger.info(
        f"Manifest loaded successfully: {len(df)} records."
    )

    return df


def validate_labels(df):

    logger.info("Validating test labels.")

    if df["label"].isna().any():

        raise DataValidationException(
            "Missing labels found."
        )

    invalid_labels = set(df["label"].unique()) - VALID_LABELS

    if invalid_labels:

        raise DataValidationException(
            f"Invalid labels found: {invalid_labels}"
        )

    logger.info("Label validation passed.")


def validate_paths(df):

    logger.info("Validating image paths.")

    missing_paths = []

    invalid_extensions = []

    for relative_path in df["image_path"]:

        image_path = PROJECT_ROOT / relative_path

        if not image_path.exists():

            missing_paths.append(relative_path)

        elif image_path.suffix.lower() not in VALID_EXTENSIONS:

            invalid_extensions.append(relative_path)

    if missing_paths:

        raise DataValidationException(
            f"Missing image files: {len(missing_paths)}"
        )

    if invalid_extensions:

        raise DataValidationException(
            f"Invalid image extensions: {len(invalid_extensions)}"
        )

    logger.info("Image path validation passed.")


def validate_image_integrity(df):

    logger.info("Validating image integrity.")

    corrupted_images = []

    for relative_path in df["image_path"]:

        image_path = PROJECT_ROOT / relative_path

        try:

            with Image.open(image_path) as image:

                image.verify()

            with Image.open(image_path) as image:

                image.load()

        except Exception:

            corrupted_images.append(relative_path)

    if corrupted_images:

        raise DataValidationException(
            f"Corrupted images found: {len(corrupted_images)}"
        )

    logger.info("Image integrity validation passed.")


def validate_no_overlap(df):

    logger.info("Checking for Training-Testing overlap.")

    if not OVERLAP_FILE.exists():

        raise DatasetNotFoundException(
            f"Overlap report not found: {OVERLAP_FILE}"
        )

    overlap_df = pd.read_csv(OVERLAP_FILE)

    affected_images = set(
        overlap_df["testing_image"].dropna()
    )

    remaining_overlap = (
        set(df["image_path"]) & affected_images
    )

    if remaining_overlap:

        raise DataValidationException(
            f"Overlap remains in clean test manifest: "
            f"{len(remaining_overlap)} images."
        )

    logger.info("No known exact Training-Testing overlap found.")


def validate_duplicates(df):

    logger.info("Checking duplicate image paths.")

    duplicate_paths = df[
        df["image_path"].duplicated(keep=False)
    ]

    if not duplicate_paths.empty:

        raise DataValidationException(
            f"Duplicate image paths found: {len(duplicate_paths)}"
        )

    logger.info("Duplicate path validation passed.")


def report_results(df):

    logger.info("Generating test manifest report.")

    print("\n========== TEST MANIFEST VALIDATION ==========")

    print("Total valid images:", len(df))

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nClass percentages:")

    print(
        (df["label"].value_counts(normalize=True) * 100)
        .round(2)
        .sort_index()
    )

    print("\nAll validation checks passed.")

    print("===============================================")


def main():

    try:

        logger.info("Starting clean test manifest validation.")

        df = load_manifest()

        validate_labels(df)

        validate_paths(df)

        validate_image_integrity(df)

        validate_duplicates(df)

        validate_no_overlap(df)

        report_results(df)

        logger.info(
            "Clean test manifest validation completed successfully."
        )

    except Exception:

        logger.exception(
            "Clean test manifest validation failed."
        )

        raise


if __name__ == "__main__":

    main()



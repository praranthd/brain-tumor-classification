from pathlib import Path

import pandas as pd

from src.logger import get_logger
from src.exception import (
    DatasetNotFoundException,
    DataValidationException
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits.csv"
)

VALID_LABELS = {
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
}

VALID_SPLITS = {
    "train",
    "validation"
}

VALID_LABELS = {
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
}


VALID_SPLITS = {
    "train",
    "validation"
}


def load_csv():

    try:
        if not CSV_FILE.exists():
            raise DatasetNotFoundException(
                f"CSV file not found: {CSV_FILE}"
            )

        df = pd.read_csv(CSV_FILE)

        if df.empty:
            raise DataValidationException(
                "CSV file is empty."
            )

        logger.info("CSV file loaded successfully.")
        logger.info(f"Total records: {len(df)}")

        return df

    except (DatasetNotFoundException, DataValidationException):
        logger.exception("CSV validation failed.")
        raise

    except Exception as error:
        logger.exception("Unable to read CSV file.")

        raise DataValidationException(
            "Failed to load dataset CSV."
        ) from error


def validate_columns(df):

    try:
        required_columns = {
            "image_path",
            "label",
            "split"
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise DataValidationException(
                f"Missing required columns: {missing_columns}"
            )

        logger.info("CSV columns validated successfully.")

    except DataValidationException:
        logger.exception("Column validation failed.")
        raise



def validate_missing_values(df):

    try:
        missing_values = df[
            ["image_path", "label", "split"]
        ].isnull().sum()

        if missing_values.sum() > 0:

            raise DataValidationException(
                f"Missing values found:\n{missing_values}"
            )

        logger.info("Missing-value validation passed.")

    except DataValidationException:
        logger.exception("Missing-value validation failed.")
        raise



def validate_labels(df):

    try:
        actual_labels = set(df["label"].unique())

        invalid_labels = actual_labels - VALID_LABELS

        if invalid_labels:
            raise DataValidationException(
                f"Invalid class labels found: {invalid_labels}"
            )

        logger.info("Class labels validated successfully.")

    except DataValidationException:
        logger.exception("Class label validation failed.")
        raise


def validate_splits(df):

    try:
        actual_splits = set(df["split"].unique())

        invalid_splits = actual_splits - VALID_SPLITS

        if invalid_splits:
            raise DataValidationException(
                f"Invalid split values found: {invalid_splits}"
            )

        logger.info("Dataset splits validated successfully.")

    except DataValidationException:
        logger.exception("Split validation failed.")
        raise



def validate_duplicate_paths(df):

    try:
        duplicate_paths = df[
            df["image_path"].duplicated(keep=False)
        ]

        if not duplicate_paths.empty:

            raise DataValidationException(
                f"Duplicate image paths found: "
                f"{len(duplicate_paths)} records"
            )

        logger.info("Duplicate image-path validation passed.")

    except DataValidationException:
        logger.exception("Duplicate path validation failed.")
        raise



def validate_class_distribution(df):

    try:
        distribution = pd.crosstab(
            df["split"],
            df["label"]
        )

        logger.info(
            f"\nClass distribution:\n{distribution}"
        )

        missing_classes = distribution.columns[
            (distribution == 0).any(axis=0)
        ].tolist()

        if missing_classes:
            raise DataValidationException(
                f"Classes missing from a split: {missing_classes}"
            )

        logger.info("Class distribution validation passed.")

    except DataValidationException:
        logger.exception("Class distribution validation failed.")
        raise



def validate_csv():

    logger.info("Starting CSV validation pipeline.")

    df = load_csv()

    validate_columns(df)

    validate_missing_values(df)

    validate_labels(df)

    validate_splits(df)

    validate_duplicate_paths(df)

    validate_class_distribution(df)

    logger.info("CSV validation pipeline completed successfully.")

    return df


if __name__ == "__main__":
    df = validate_csv()
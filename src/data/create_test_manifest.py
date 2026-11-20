
from pathlib import Path

import pandas as pd

from src.logger import get_logger
from src.exception import (
    DatasetNotFoundException,
    DataValidationException
)


logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TESTING_DIR = PROJECT_ROOT / "data" / "raw" / "Testing"

OVERLAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_reports"
    / "train_test_overlap.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_manifest.csv"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_testing_images():

    logger.info("Collecting original Testing images.")

    if not TESTING_DIR.exists():

        raise DatasetNotFoundException(
            f"Testing directory not found: {TESTING_DIR}"
        )

    records = []

    for class_name in CLASS_NAMES:

        class_dir = TESTING_DIR / class_name

        if not class_dir.exists():

            raise DatasetNotFoundException(
                f"Class directory not found: {class_dir}"
            )

        for image_path in class_dir.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower() in VALID_EXTENSIONS
            ):

                records.append({
                    "image_path": str(
                        image_path.relative_to(PROJECT_ROOT)
                    ),
                    "label": class_name
                })

    df = pd.DataFrame(records)

    if df.empty:

        raise DataValidationException(
            "No Testing images were found."
        )

    logger.info(
        f"Total Testing images collected: {len(df)}"
    )

    return df


def load_overlap_report():

    logger.info("Loading Training-Testing overlap report.")

    if not OVERLAP_FILE.exists():

        raise DatasetNotFoundException(
            f"Overlap report not found: {OVERLAP_FILE}"
        )

    overlap_df = pd.read_csv(OVERLAP_FILE)

    if "testing_image" not in overlap_df.columns:

        raise DataValidationException(
            "Overlap report does not contain testing_image column."
        )

    affected_images = set(
        overlap_df["testing_image"].dropna()
    )

    logger.info(
        f"Unique affected Testing images: {len(affected_images)}"
    )

    return affected_images


def create_clean_test_manifest():

    logger.info("Starting clean Testing manifest creation.")

    testing_df = collect_testing_images()

    affected_images = load_overlap_report()

    # Keep only Testing images that do not overlap.
    clean_df = testing_df[
        ~testing_df["image_path"].isin(affected_images)
    ].copy()

    if clean_df.empty:

        raise DataValidationException(
            "Clean Testing manifest is empty."
        )

    clean_df = clean_df.reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    clean_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    logger.info(
        f"Clean test manifest saved: {OUTPUT_FILE}"
    )

    return testing_df, clean_df, affected_images


def report_results(testing_df, clean_df, affected_images):

    excluded_count = len(testing_df) - len(clean_df)

    logger.info("=" * 55)
    logger.info("CLEAN TEST MANIFEST REPORT")
    logger.info("=" * 55)

    print("\n========== CLEAN TEST MANIFEST REPORT ==========")

    print("\nOriginal Testing images:", len(testing_df))

    print("Affected Testing images:", len(affected_images))

    print("Excluded Testing images:", excluded_count)

    print("Clean Testing images:", len(clean_df))

    print("\nOriginal class distribution:")

    print(
        testing_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nClean Testing class distribution:")

    print(
        clean_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\n=================================================")

    logger.info("Clean test manifest creation completed.")


def main():

    try:

        testing_df, clean_df, affected_images = (
            create_clean_test_manifest()
        )

        report_results(
            testing_df,
            clean_df,
            affected_images
        )

    except Exception:

        logger.exception(
            "Clean test manifest creation failed."
        )

        raise


if __name__ == "__main__":

    main()

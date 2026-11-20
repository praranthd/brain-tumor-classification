# from pathlib import Path
# from collections import defaultdict
# import hashlib
# import numpy as np
# import pandas as pd

# from PIL import Image

# from src.logger import get_logger
# from src.exception import (
#     DatasetNotFoundException,
#     DataValidationException
# )

# from src.data.data_validation.validate_csv import validate_csv

# logger = get_logger(__name__)

# PROJECT_ROOT = Path(__file__).resolve().parents[3]

# TRAINING_DIR = PROJECT_ROOT / "data" / "raw" / "Training"

# TESTING_DIR = PROJECT_ROOT / "data" / "raw" / "Testing"


# REPORT_DIR = (
#     PROJECT_ROOT
#     / "data"
#     / "processed"
#     / "validation_reports"
# )

# NEAR_DUPLICATE_THRESHOLD = 5


# def calculate_image_hash(image_path):

#     try:
#         with Image.open(image_path) as image:

#             image = image.convert("RGB")

#             image_hash = hashlib.sha256()

#             image_hash.update(str(image.size).encode())

#             image_hash.update(image.tobytes())

#             return image_hash.hexdigest()

#     except Exception as error:

#         logger.exception(
#             f"Failed to calculate image hash: {image_path}"
#         )

#         raise DataValidationException(
#             f"Image hashing failed: {image_path}"
#         ) from error


# def collect_image_hashes(dataset_dir, dataset_name):

#     image_hashes = defaultdict(list)

#     try:

#         if not dataset_dir.exists():
#             raise DatasetNotFoundException(
#                 f"{dataset_name} directory not found: {dataset_dir}"
#             )

#         class_names = [
#             "glioma",
#             "meningioma",
#             "notumor",
#             "pituitary"
#         ]

#         valid_extensions = {".jpg", ".jpeg", ".png"}

#         for class_name in class_names:

#             class_dir = dataset_dir / class_name

#             if not class_dir.exists():
#                 raise DatasetNotFoundException(
#                     f"Class directory not found: {class_dir}"
#                 )

#             for image_path in class_dir.iterdir():

#                 if (
#                     image_path.is_file()
#                     and image_path.suffix.lower() in valid_extensions
#                 ):

#                     image_hash = calculate_image_hash(image_path)

#                     image_hashes[image_hash].append({
#                         "image_path": str(
#                             image_path.relative_to(PROJECT_ROOT)
#                         ),
#                         "label": class_name,
#                         "dataset": dataset_name
#                     })

#         logger.info(
#             f"{dataset_name}: {sum(map(len, image_hashes.values()))} "
#             f"images hashed successfully."
#         )

#         return image_hashes

#     except Exception as error:

#         logger.exception(
#             f"Failed to collect image hashes for {dataset_name}."
#         )

#         raise DataValidationException(
#             f"Failed to collect image hashes for {dataset_name}."
#         ) from error


# def detect_train_test_overlap():

#     try:

#         logger.info("Starting Training-Testing overlap audit.")

#         training_hashes = collect_image_hashes(
#             TRAINING_DIR,
#             "Training"
#         )

#         testing_hashes = collect_image_hashes(
#             TESTING_DIR,
#             "Testing"
#         )

#         overlap_records = []

#         common_hashes = (
#             set(training_hashes.keys())
#             & set(testing_hashes.keys())
#         )

#         for image_hash in common_hashes:

#             for train_record in training_hashes[image_hash]:

#                 for test_record in testing_hashes[image_hash]:

#                     overlap_records.append({
#                         "hash": image_hash,
#                         "training_image": train_record["image_path"],
#                         "training_label": train_record["label"],
#                         "testing_image": test_record["image_path"],
#                         "testing_label": test_record["label"],
#                         "same_label": (
#                             train_record["label"]
#                             == test_record["label"]
#                         )
#                     })

#         overlap_df = pd.DataFrame(overlap_records)

#         REPORT_DIR.mkdir(parents=True, exist_ok=True)

#         report_file = (
#             REPORT_DIR / "train_test_overlap.csv"
#         )

#         overlap_df.to_csv(
#             report_file,
#             index=False
#         )

#         if not overlap_df.empty:

#             logger.warning(
#                 f"Training-Testing exact overlap found: "
#                 f"{len(overlap_df)} matching pairs."
#             )

#         else:

#             logger.info(
#                 "No exact image overlap found between Training and Testing."
#             )

#         logger.info(
#             f"Training-Testing overlap report saved: {report_file}"
#         )

#         return overlap_df

#     except Exception as error:

#         logger.exception(
#             "Training-Testing overlap detection failed."
#         )

#         raise DataValidationException(
#             "Training-Testing overlap detection failed."
#         ) from error





# def detect_exact_duplicates(df):

#     hash_groups = defaultdict(list)

#     try:
#         for _, row in df.iterrows():

#             image_path = PROJECT_ROOT / row["image_path"]

#             image_hash = calculate_image_hash(image_path)

#             hash_groups[image_hash].append({
#                 "image_path": row["image_path"],
#                 "label": row["label"],
#                 "split": row["split"]
#             })

#         duplicates = []

#         for image_hash, records in hash_groups.items():

#             if len(records) > 1:

#                 for record in records:
#                     duplicates.append({
#                         "hash": image_hash,
#                         "image_path": record["image_path"],
#                         "label": record["label"],
#                         "split": record["split"],
#                         "duplicate_count": len(records)
#                     })

#         REPORT_DIR.mkdir(parents=True, exist_ok=True)

#         report_file = REPORT_DIR / "exact_duplicates.csv"

#         duplicate_df = pd.DataFrame(duplicates)

#         duplicate_df.to_csv(report_file, index=False)

#         if duplicates:

#             logger.warning(
#                 f"Exact duplicate images found: {len(duplicates)} records"
#             )

#         else:

#             logger.info("No exact duplicate images found.")

#         logger.info(f"Duplicate report saved: {report_file}")

#         return duplicate_df

#     except Exception as error:

#         logger.exception("Exact duplicate detection failed.")

#         raise DataValidationException(
#             "Exact duplicate detection failed."
#         ) from error


# def detect_split_leakage(duplicate_df):

#     try:
#         leakage_records = []

#         if duplicate_df.empty:
#             logger.info("No exact duplicates available for leakage analysis.")
#             return pd.DataFrame()

#         grouped = duplicate_df.groupby("hash")

#         for image_hash, group in grouped:

#             splits = group["split"].unique()

#             if len(splits) > 1:

#                 leakage_records.append({
#                     "hash": image_hash,
#                     "image_paths": " | ".join(
#                         group["image_path"].tolist()
#                     ),
#                     "labels": " | ".join(
#                         group["label"].astype(str).tolist()
#                     ),
#                     "splits": " | ".join(splits),
#                     "record_count": len(group)
#                 })

#         leakage_df = pd.DataFrame(leakage_records)

#         REPORT_DIR.mkdir(parents=True, exist_ok=True)

#         report_file = REPORT_DIR / "split_leakage.csv"

#         leakage_df.to_csv(report_file, index=False)

#         if not leakage_df.empty:

#             logger.warning(
#                 f"Train-validation leakage groups found: "
#                 f"{len(leakage_df)}"
#             )

#         else:

#             logger.info(
#                 "No exact duplicate leakage between splits found."
#             )

#         logger.info(f"Leakage report saved: {report_file}")

#         return leakage_df

#     except Exception as error:

#         logger.exception("Split leakage detection failed.")

#         raise DataValidationException(
#             "Split leakage detection failed."
#         ) from error



# if __name__ == "__main__":

#     df = validate_csv()

#     duplicates = detect_exact_duplicates(df)

#     leakage = detect_split_leakage(duplicates)

#     # Check whether identical images have different labels
#     label_conflicts = (
#         duplicates.groupby("hash")["label"]
#         .nunique()
#     )

#     conflicting_hashes = label_conflicts[label_conflicts > 1]

#     print("\nExact duplicate records:", len(duplicates))
#     print("Train-validation leakage groups:", len(leakage))
#     print("Conflicting label groups:", len(conflicting_hashes))
#     train_test_overlap = detect_train_test_overlap()

#     print("\nTraining-Testing exact overlap pairs:", len(train_test_overlap))

#     if not conflicting_hashes.empty:
#         print("\nConflicting records:")
#         print(
#             duplicates[
#                 duplicates["hash"].isin(conflicting_hashes.index)
#             ].to_string(index=False)
#         )
#     else:
#         print("\nNo conflicting labels found.")




from pathlib import Path
from collections import defaultdict
import hashlib

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

TRAINING_DIR = PROJECT_ROOT / "data" / "raw" / "Training"
TESTING_DIR = PROJECT_ROOT / "data" / "raw" / "Testing"

SPLIT_FILE = PROJECT_ROOT / "data" / "processed" / "splits.csv"

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_reports"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------
# 1. Calculate image hash
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Detect exact duplicates in split manifest
# --------------------------------------------------

def detect_exact_duplicates(df):

    logger.info("Starting exact duplicate detection.")

    hash_records = defaultdict(list)

    for _, row in df.iterrows():

        image_path = PROJECT_ROOT / row["image_path"]

        image_hash = calculate_image_hash(image_path)

        hash_records[image_hash].append({
            "image_path": row["image_path"],
            "label": row["label"],
            "split": row["split"],
            "hash": image_hash
        })

    duplicate_records = []

    for image_hash, records in hash_records.items():

        if len(records) > 1:

            duplicate_records.extend(records)

    duplicate_df = pd.DataFrame(duplicate_records)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_file = REPORT_DIR / "exact_duplicates.csv"

    duplicate_df.to_csv(report_file, index=False)

    logger.info(
        f"Exact duplicate records found: {len(duplicate_df)}"
    )

    logger.info(
        f"Exact duplicate report saved: {report_file}"
    )

    return duplicate_df


# --------------------------------------------------
# 3. Detect train-validation leakage
# --------------------------------------------------

def detect_split_leakage(duplicate_df):

    logger.info("Checking train-validation leakage.")

    if duplicate_df.empty:

        logger.info("No duplicate groups found.")

        return pd.DataFrame()

    leakage_records = []

    for image_hash, group in duplicate_df.groupby("hash"):

        splits = set(group["split"])

        if "train" in splits and "validation" in splits:

            leakage_records.extend(
                group.to_dict(orient="records")
            )

    leakage_df = pd.DataFrame(leakage_records)

    report_file = REPORT_DIR / "split_leakage.csv"

    leakage_df.to_csv(report_file, index=False)

    if leakage_df.empty:

        logger.info(
            "No exact duplicate leakage between train and validation."
        )

    else:

        logger.warning(
            f"Train-validation leakage records: {len(leakage_df)}"
        )

    return leakage_df


# --------------------------------------------------
# 4. Collect hashes from Training or Testing folder
# --------------------------------------------------

def collect_image_hashes(dataset_dir, dataset_name):

    image_hashes = defaultdict(list)

    if not dataset_dir.exists():

        raise DatasetNotFoundException(
            f"{dataset_name} directory not found: {dataset_dir}"
        )

    for class_name in CLASS_NAMES:

        class_dir = dataset_dir / class_name

        if not class_dir.exists():

            raise DatasetNotFoundException(
                f"Class directory not found: {class_dir}"
            )

        for image_path in class_dir.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower() in VALID_EXTENSIONS
            ):

                image_hash = calculate_image_hash(image_path)

                image_hashes[image_hash].append({
                    "image_path": str(
                        image_path.relative_to(PROJECT_ROOT)
                    ),
                    "label": class_name,
                    "dataset": dataset_name
                })

    logger.info(
        f"{dataset_name}: "
        f"{sum(map(len, image_hashes.values()))} images hashed."
    )

    return image_hashes


# --------------------------------------------------
# 5. Detect Training-Testing overlap
# --------------------------------------------------

def detect_train_test_overlap():

    logger.info("Starting Training-Testing overlap audit.")

    training_hashes = collect_image_hashes(
        TRAINING_DIR,
        "Training"
    )

    testing_hashes = collect_image_hashes(
        TESTING_DIR,
        "Testing"
    )

    overlap_records = []

    common_hashes = (
        set(training_hashes.keys())
        & set(testing_hashes.keys())
    )

    for image_hash in common_hashes:

        for train_record in training_hashes[image_hash]:

            for test_record in testing_hashes[image_hash]:

                overlap_records.append({

                    "hash": image_hash,

                    "training_image": train_record["image_path"],

                    "training_label": train_record["label"],

                    "testing_image": test_record["image_path"],

                    "testing_label": test_record["label"],

                    "same_label": (
                        train_record["label"]
                        == test_record["label"]
                    )

                })

    overlap_df = pd.DataFrame(overlap_records)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_file = REPORT_DIR / "train_test_overlap.csv"

    overlap_df.to_csv(report_file, index=False)

    if overlap_df.empty:

        logger.info(
            "No exact image overlap found between Training and Testing."
        )

    else:

        logger.warning(
            f"Training-Testing exact overlap pairs: {len(overlap_df)}"
        )

    logger.info(
        f"Training-Testing overlap report saved: {report_file}"
    )

    return overlap_df


# --------------------------------------------------
# 6. Identify affected Testing images by split
# --------------------------------------------------

def analyze_overlap_by_split(overlap_df):

    logger.info("Analyzing overlap by training split.")

    if overlap_df.empty:

        logger.info("No overlap to analyze.")

        return pd.DataFrame()

    if not SPLIT_FILE.exists():

        raise DatasetNotFoundException(
            f"Split manifest not found: {SPLIT_FILE}"
        )

    split_df = pd.read_csv(SPLIT_FILE)

    required_columns = {"image_path", "split"}

    if not required_columns.issubset(split_df.columns):

        raise DataValidationException(
            "Split manifest must contain image_path and split columns."
        )

    # Remove repeated Training-Testing pairs before joining.
    unique_pairs = overlap_df[
        ["training_image", "testing_image", "hash"]
    ].drop_duplicates()

    # Attach train/validation information to Training images.
    merged_df = unique_pairs.merge(
        split_df[["image_path", "split"]],
        left_on="training_image",
        right_on="image_path",
        how="left"
    )

    if merged_df["split"].isna().any():

        raise DataValidationException(
            "Some overlapping Training images are missing from splits.csv."
        )

    # Count distinct Testing images for each split.
    summary = (
        merged_df.groupby("split")
        .agg(
            unique_testing_images=("testing_image", "nunique"),
            unique_hashes=("hash", "nunique")
        )
        .reset_index()
    )

    report_file = REPORT_DIR / "overlap_by_split.csv"

    summary.to_csv(report_file, index=False)

    logger.info(
        f"Overlap-by-split report saved: {report_file}"
    )

    return summary


# --------------------------------------------------
# 7. Main execution
# --------------------------------------------------

def main():

    try:

        logger.info("=" * 60)
        logger.info("STARTING DATA DUPLICATE VALIDATION")
        logger.info("=" * 60)

        # Validate the existing split CSV.
        df = validate_csv()

        # Training pool duplicate detection.
        duplicates = detect_exact_duplicates(df)

        # Train-validation leakage detection.
        leakage = detect_split_leakage(duplicates)

        # Training-Testing overlap detection.
        train_test_overlap = detect_train_test_overlap()

        # Analyze which Training split overlaps with Testing.
        overlap_summary = analyze_overlap_by_split(
            train_test_overlap
        )

        # Find conflicting labels inside duplicate groups.
        if duplicates.empty:

            conflicting_hashes = pd.Series(dtype=object)

        else:

            label_counts = (
                duplicates.groupby("hash")["label"].nunique()
            )

            conflicting_hashes = label_counts[
                label_counts > 1
            ]

        # Training-Testing label conflicts.
        if train_test_overlap.empty:

            train_test_conflicts = pd.DataFrame()

        else:

            train_test_conflicts = train_test_overlap[
                ~train_test_overlap["same_label"]
            ]

        # Summary.
        print("\n========== DUPLICATE VALIDATION REPORT ==========")

        print(
            "\nExact duplicate records:",
            len(duplicates)
        )

        print(
            "Train-validation leakage groups:",
            leakage["hash"].nunique() if not leakage.empty else 0
        )

        print(
            "Conflicting label groups within Training pool:",
            len(conflicting_hashes)
        )

        print(
            "\nTraining-Testing matching pairs:",
            len(train_test_overlap)
        )

        print(
            "Unique overlapping hashes:",
            train_test_overlap["hash"].nunique()
            if not train_test_overlap.empty else 0
        )

        print(
            "Unique affected Testing images:",
            train_test_overlap["testing_image"].nunique()
            if not train_test_overlap.empty else 0
        )

        print(
            "Unique affected Training images:",
            train_test_overlap["training_image"].nunique()
            if not train_test_overlap.empty else 0
        )

        print(
            "Training-Testing conflicting label pairs:",
            len(train_test_conflicts)
        )

        if not overlap_summary.empty:

            print("\nOverlap by Training split:")

            print(overlap_summary.to_string(index=False))

        print("\n=================================================")

        logger.info("Duplicate validation completed successfully.")

    except Exception:

        logger.exception(
            "Duplicate validation pipeline failed."
        )

        raise


if __name__ == "__main__":

    main()

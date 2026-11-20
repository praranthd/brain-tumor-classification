from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.logger import get_logger
from src.exception import (
    DatasetNotFoundException,
    InvalidImageException,
    DataValidationException
)

from src.data.data_validation.validate_csv import validate_csv

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]



def validate_image_paths(df):

    missing_images = []

    try:
        for image_path in df["image_path"]:

            full_path = PROJECT_ROOT / image_path

            if not full_path.exists():
                missing_images.append(str(image_path))

        if missing_images:

            logger.error(
                f"Missing images found: {len(missing_images)}"
            )

            raise DatasetNotFoundException(
                f"Missing image paths: {missing_images[:10]}"
            )

        logger.info(
            f"Image existence validation passed. "
            f"Total images checked: {len(df)}"
        )

    except DatasetNotFoundException:
        logger.exception("Image existence validation failed.")
        raise


def validate_image_integrity(df):

    corrupted_images = []

    try:
        for image_path in df["image_path"]:

            full_path = PROJECT_ROOT / image_path

            try:
                with Image.open(full_path) as image:
                    image.verify()

            except (
                UnidentifiedImageError,
                OSError,
                SyntaxError,
                ValueError
            ):
                corrupted_images.append(str(image_path))

        if corrupted_images:

            logger.error(
                f"Corrupted images found: {len(corrupted_images)}"
            )

            raise InvalidImageException(
                f"Corrupted image paths: {corrupted_images[:10]}"
            )

        logger.info(
            f"Image integrity validation passed. "
            f"Total images checked: {len(df)}"
        )

    except InvalidImageException:
        logger.exception("Image integrity validation failed.")
        raise

def validate_image_decoding(df):

    invalid_images = []

    try:
        for image_path in df["image_path"]:

            full_path = PROJECT_ROOT / image_path

            try:
                with Image.open(full_path) as image:
                    image.load()
                    image.convert("RGB")

            except (
                UnidentifiedImageError,
                OSError,
                SyntaxError,
                ValueError
            ):
                invalid_images.append(str(image_path))

        if invalid_images:

            logger.error(
                f"Unreadable images found: {len(invalid_images)}"
            )

            raise InvalidImageException(
                f"Unreadable image paths: {invalid_images[:10]}"
            )

        logger.info(
            f"Image decoding validation passed. "
            f"Total images checked: {len(df)}"
        )

    except InvalidImageException:
        logger.exception("Image decoding validation failed.")
        raise

def validate_images():

    logger.info("Starting image validation pipeline.")

    df = validate_csv()

    validate_image_paths(df)

    validate_image_integrity(df)

    validate_image_decoding(df)

    logger.info("Image validation pipeline completed successfully.")

    return True



if __name__ == "__main__":
    validate_images()
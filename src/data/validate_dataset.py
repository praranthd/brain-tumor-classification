from pathlib import Path

from src.logger import get_logger
from src.exception import DatasetNotFoundException

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = PROJECT_ROOT / "data" / "processed" / "splits.csv"


def validate_csv_file():

    try:
        if not CSV_FILE.exists():
            raise DatasetNotFoundException(
                f"Dataset CSV not found: {CSV_FILE}"
            )

        logger.info("Dataset CSV file found successfully.")

        return CSV_FILE

    except DatasetNotFoundException:
        logger.exception("Dataset CSV validation failed.")
        raise


if __name__ == "__main__":
    validate_csv_file()
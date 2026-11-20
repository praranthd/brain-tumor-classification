
import logging
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = LOG_DIR / "project.log"


def get_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Common log format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger

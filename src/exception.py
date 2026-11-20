
from src.logger import get_logger

logger = get_logger("BrainTumorProject")


class BrainTumorException(Exception):
    """Base exception for the Brain Tumor project."""
    pass


class DatasetNotFoundException(BrainTumorException):
    pass


class InvalidImageException(BrainTumorException):
    pass


class DataValidationException(BrainTumorException):
    pass


class ConfigurationException(BrainTumorException):
    pass


def handle_exception(error: Exception, message: str):
    """
    Log the error and raise a project-specific exception.
    """
    logger.exception(message)
    raise BrainTumorException(message) from error

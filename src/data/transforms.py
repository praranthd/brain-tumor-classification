
from torchvision import transforms
from PIL import Image, ImageOps

from src.logger import get_logger
from src.exception import (
    InvalidImageException,
    ConfigurationException
)

logger = get_logger(__name__)


class ResizeWithPadding:

    def __init__(self, image_size=224):

        if not isinstance(image_size, int) or image_size <= 0:
            raise ConfigurationException(
                f"Invalid image size: {image_size}. "
                "Image size must be a positive integer."
            )

        self.image_size = image_size

        logger.info(
            f"ResizeWithPadding initialized: {image_size}x{image_size}"
        )

    def __call__(self, image):

        try:

            if not isinstance(image, Image.Image):
                raise TypeError("Input must be a PIL Image.")

            image = ImageOps.pad(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.BILINEAR,
                color=(0, 0, 0),
                centering=(0.5, 0.5)
            )

            return image

        except Exception as error:

            logger.exception("Image resizing and padding failed.")

            raise InvalidImageException(
                "Failed to resize and pad MRI image."
            ) from error


def get_train_transforms(image_size=224):

    try:

        train_transform = transforms.Compose([

            ResizeWithPadding(image_size),

            transforms.RandomRotation(degrees=10),

            transforms.ToTensor()

        ])

        logger.info("Training transformations initialized.")

        return train_transform

    except Exception as error:

        logger.exception(
            "Failed to initialize training transformations."
        )

        raise ConfigurationException(
            "Training transformation configuration failed."
        ) from error


def get_validation_transforms(image_size=224):

    try:

        validation_transform = transforms.Compose([

            ResizeWithPadding(image_size),

            transforms.ToTensor()

        ])

        logger.info("Validation transformations initialized.")

        return validation_transform

    except Exception as error:

        logger.exception(
            "Failed to initialize validation transformations."
        )

        raise ConfigurationException(
            "Validation transformation configuration failed."
        ) from error

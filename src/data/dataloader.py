
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import BrainMRIDataset
from src.data.transforms import (
    get_train_transforms,
    get_validation_transforms
)

from src.logger import get_logger
from src.exception import (
    ConfigurationException,
    DataValidationException
)

logger = get_logger(__name__)


def create_dataloaders(
    csv_file,
    root_dir,
    batch_size=32,
    num_workers=0,
    image_size=224
):

    try:

        csv_file = Path(csv_file)
        root_dir = Path(root_dir)

        # Validate configuration
        if batch_size <= 0:
            raise ConfigurationException(
                "Batch size must be greater than zero."
            )

        if num_workers < 0:
            raise ConfigurationException(
                "Number of workers cannot be negative."
            )

        if image_size <= 0:
            raise ConfigurationException(
                "Image size must be greater than zero."
            )

        # Create training dataset
        train_dataset = BrainMRIDataset(
            csv_file=csv_file,
            root_dir=root_dir,
            split="train",
            transform=get_train_transforms(image_size)
        )

        # Create validation dataset
        validation_dataset = BrainMRIDataset(
            csv_file=csv_file,
            root_dir=root_dir,
            split="validation",
            transform=get_validation_transforms(image_size)
        )

        # Create training DataLoader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False
        )

        # Create validation DataLoader
        validation_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False
        )

        logger.info("DataLoaders created successfully.")

        logger.info(
            f"Training batches: {len(train_loader)}"
        )

        logger.info(
            f"Validation batches: {len(validation_loader)}"
        )

        return train_loader, validation_loader

    except (ConfigurationException, DataValidationException):
        raise

    except Exception as error:

        logger.exception("Failed to create DataLoaders.")

        raise ConfigurationException(
            "DataLoader initialization failed."
        ) from error

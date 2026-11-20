
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

from src.logger import get_logger
from src.exception import (
    InvalidImageException,
    DataValidationException,
    DatasetNotFoundException
)

logger = get_logger(__name__)


class BrainMRIDataset(Dataset):

    LABEL_MAP = {
        "glioma": 0,
        "meningioma": 1,
        "notumor": 2,
        "pituitary": 3
    }

    VALID_SPLITS = {"train", "validation"}

    def __init__(
        self,
        csv_file,
        root_dir,
        split,
        transform=None
    ):

        self.csv_file = Path(csv_file)
        self.root_dir = Path(root_dir)
        self.split = split

        # Validate split
        if self.split not in self.VALID_SPLITS:
            raise DataValidationException(
                f"Invalid split: {self.split}. "
                f"Choose from {self.VALID_SPLITS}"
            )

        # Validate CSV existence
        if not self.csv_file.exists():
            raise DatasetNotFoundException(
                f"Dataset CSV not found: {self.csv_file}"
            )

        try:
            self.data = pd.read_csv(self.csv_file)

        except Exception as error:
            logger.exception("Failed to read dataset CSV.")

            raise DataValidationException(
                "Unable to read dataset CSV."
            ) from error

        # Validate required columns
        required_columns = {
            "image_path",
            "label",
            "split"
        }

        if not required_columns.issubset(self.data.columns):
            raise DataValidationException(
                f"CSV must contain columns: {required_columns}"
            )

        # Filter training or validation records
        self.data = self.data[
            self.data["split"] == self.split
        ].reset_index(drop=True)

        if self.data.empty:
            raise DataValidationException(
                f"No records found for split: {self.split}"
            )

        # Validate labels
        unknown_labels = (
            set(self.data["label"].dropna())
            - set(self.LABEL_MAP)
        )

        if unknown_labels:
            raise DataValidationException(
                f"Unknown labels found: {unknown_labels}"
            )

        if self.data["label"].isna().any():
            raise DataValidationException(
                "Missing labels found in dataset."
            )

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        logger.info(
            f"{self.split} dataset initialized "
            f"with {len(self.data)} images."
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = self.root_dir / row["image_path"]

        label = self.LABEL_MAP[row["label"]]

        try:

            with Image.open(image_path) as img:
                image = img.convert("RGB")

            if self.transform:
                image = self.transform(image)

        except (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
            ValueError
        ) as error:

            logger.exception(
                f"Unable to process image: {image_path}"
            )

            raise InvalidImageException(
                f"Invalid or missing image: {image_path}"
            ) from error

        except Exception as error:

            logger.exception(
                f"Unexpected error processing image: {image_path}"
            )

            raise InvalidImageException(
                f"Image processing failed: {image_path}"
            ) from error

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label

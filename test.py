
from pathlib import Path

from src.data.dataloader import create_dataloaders

PROJECT_ROOT = Path(__file__).resolve().parent

CSV_FILE = (
    PROJECT_ROOT / "data" / "processed" / "splits.csv"
)

train_loader, validation_loader = create_dataloaders(
    csv_file=CSV_FILE,
    root_dir=PROJECT_ROOT,
    batch_size=32,
    num_workers=0,
    image_size=224
)

print("\n========== DATALOADER REPORT ==========")

print("Training batches:", len(train_loader))
print("Validation batches:", len(validation_loader))

# Test one training batch
images, labels = next(iter(train_loader))

print("\n========== TRAINING BATCH ==========")

print("Image shape:", images.shape)
print("Label shape:", labels.shape)

# Test one validation batch
images, labels = next(iter(validation_loader))

print("\n========== VALIDATION BATCH ==========")

print("Image shape:", images.shape)
print("Label shape:", labels.shape)

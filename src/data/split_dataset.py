
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_DIR = PROJECT_ROOT / "data" / "raw" / "Training"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "splits.csv"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

records = []

# Collect image paths and labels
for class_name in CLASSES:

    class_dir = TRAINING_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    for image_path in class_dir.iterdir():

        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:

            records.append({
                "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                "label": class_name
            })


# Create DataFrame
df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No images found in the training dataset.")

# Stratified train-validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Assign split names
train_df["split"] = "train"
val_df["split"] = "validation"

# Combine both datasets
split_df = pd.concat([train_df, val_df], ignore_index=True)

# Save split manifest
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
split_df.to_csv(OUTPUT_FILE, index=False)

# Report
print("Dataset splitting completed.")
print(f"Total images: {len(split_df)}")
print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"\nClass distribution:\n")
print(split_df.groupby(["split", "label"]).size())
print(f"\nSplit file saved to: {OUTPUT_FILE}")

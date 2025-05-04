import os
import random
from pathlib import Path
from collections import defaultdict
import shutil

def split_text_file_by_class(
    data_file: Path,
    output_dir = None,
    train_ratio: float = 0.8
) -> None:
    """
    Read lines of 'filename,label', group by label, and split into train/val sets.

    Args:
        data_file (Path): Path to the input text file with 'name,label' per line.
        output_dir (Path): Directory to save 'train.txt' and 'val.txt'.
        train_ratio (float): Proportion of data to use for training (default 0.8).
    """
    data_file = Path(data_file)
    # Read and group entries by class label
    data_by_class = defaultdict(list)
    for line in data_file.read_text().splitlines():
        if not line.strip():
            continue
        name, label = line.split(',', 1)
        data_by_class[label].append(name)

    if output_dir is None:
        output_dir = data_file.parent
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    train_lines, val_lines = [], []
    # Shuffle and split per class
    for label, names in data_by_class.items():
        random.shuffle(names)
        split_idx = int(len(names) * train_ratio)
        # Prepare train and val entries
        train_lines.extend(f"{n},{label}" for n in names[:split_idx])
        val_lines.extend(f"{n},{label}" for n in names[split_idx:])

    # Write the split files
    (output_dir / 'train.txt').write_text("\n".join(train_lines))
    (output_dir / 'val.txt').write_text("\n".join(val_lines))

    print(
        f"Split complete: {len(train_lines)} training and {len(val_lines)} validation samples saved to '{output_dir}'"
    )

def split_dataset_folder(
    data_dir: Path,
    output_dir = None,
    train_ratio: float = 0.8
) -> None:
    """
    Split a folder-structured dataset into training and validation sets.

    Args:
        data_dir (str or Path): Path to the root dataset directory where each subfolder is a class.
        output_dir (str or Path): Path to the directory where 'train' and 'val' folders will be created.
        val_ratio (float): Proportion of images per class to use for validation (default 0.2).
    """
    # Convert input paths to Path objects
    data_path = Path(data_dir)

    if output_dir is None:
        output_dir = data_path.parent
    # Define train and validation directories
    train_path = Path(output_dir) / "train"
    val_path = Path(output_dir) / "val"

    # Create base output directories
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    total_train_images = total_val_images = 0

    # Iterate over each class folder
    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        # List image files in the class folder
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in image_exts]
        random.shuffle(images)

        # Compute split index
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create class-specific subdirectories in train and val
        train_class_dir = train_path / class_name
        val_class_dir = val_path / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # Copy training images
        for img in train_images:
            shutil.copy(img, train_class_dir / img.name)
        # Copy validation images
        for img in val_images:
            shutil.copy(img, val_class_dir / img.name)

        total_train_images += len(train_images)
        total_val_images += len(val_images)

        # Print summary for this class
        print(f"Class '{class_name}': total={len(images)}, "f"train={len(train_images)}, val={len(val_images)}")

    # Final summary
    print(
        f"\nDataset split complete.\n"
        f"Training samples are in: {train_path}, train num is {total_train_images}\n"
        f"Validation samples are in: {val_path}, val num is {total_val_images}\n"
    )

if __name__ == "__main__":
    # folder
    # data_dir = "/Users/yiyezhetian/Downloads/mini-imagenet"
    # split_dataset_folder(data_dir)

    # image_info
    data_file = "data_info.txt"
    split_text_file_by_class(data_file)

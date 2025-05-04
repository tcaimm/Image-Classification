import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def compute_dataset_mean_std_with_progress(root_dir, exts={'.jpg', '.jpeg', '.png', '.bmp'}):
    """
    Walk through root_dir and all its subdirectories, compute per-channel RGB mean and std.

    Steps:
    1. Collect all image file paths under root_dir.
    2. Iterate with a tqdm progress bar.
    3. Convert each image to RGB and normalize pixel values to [0,1].
    4. Accumulate channel sums and sums of squares.
    5. Compute final mean and std per channel.

    Args:
        root_dir (str): Path to the root directory of the image dataset.
        exts (set of str): Allowed image file extensions (lowercase).

    Returns:
        mean (np.ndarray): Array of shape (3,) with R, G, B channel means.
        std  (np.ndarray): Array of shape (3,) with R, G, B channel standard deviations.
    """
    # STEP 1: Gather all image file paths
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in exts:
                image_paths.append(os.path.join(dirpath, fname))

    # Initialize accumulators
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    # STEP 2: Process each image with a progress bar
    for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
        # STEP 3: Load and convert to RGB
        img = Image.open(img_path).convert('RGB')
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to [0,1]
        h, w, _ = arr.shape
        pixels = h * w

        # STEP 4: Accumulate sums
        flat = arr.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sum_sq += (flat ** 2).sum(axis=0)
        total_pixels += pixels

    # STEP 5: Compute mean and std
    mean = channel_sum / total_pixels
    # variance = E[x^2] - (E[x])^2
    var = channel_sum_sq / total_pixels - mean ** 2
    std = np.sqrt(var)

    return mean, std


if __name__ == '__main__':
    dataset_root = ''
    mean, std = compute_dataset_mean_std_with_progress(dataset_root)
    print(f"RGB mean: {mean.tolist()}")
    print(f"RGB std : {std.tolist()}")


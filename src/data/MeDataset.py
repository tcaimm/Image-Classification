import os
import json
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class ImageClassificationDataset(Dataset):
    """
    A flexible dataset for image classification that supports:
    1. Text mode: label mappings provided in a text file.
    2. Folder mode: each class is a subfolder containing images.
    """

    def __init__(self,
                 image_root_dir=None,
                 image_info_file=None,
                 label2index_json=None,
                 transform=None,
                 mode="text"):
        """
        Args:
            image_root_dir (str): Root directory containing images or class folders.
            image_info_file (str): Path to a text file with image names and labels (used in 'text' mode).
            label2index_json (str): JSON file mapping class names to integer labels (only in 'text' mode).
            transform (callable): Optional transform to be applied on a sample.
            mode (str): Either 'text' or 'folder' to determine the loading strategy.
        """
        self.image_root_dir = image_root_dir
        self.image_info_file = image_info_file
        self.label2index_json = label2index_json
        self.transform = transform
        self.mode = mode.lower()
        self.image_names = []
        self.image_labels = []
        self.label_to_index = {}

        if self.mode == "text":
            self._load_from_text_file()
        elif self.mode == "folder":
            self._load_from_folder_structure()
        else:
            raise ValueError("Mode must be either 'text' or 'folder'.")

    @property
    def index_to_label(self):
        return {v: k for k, v in self.label_to_index.items()}

    def _load_from_text_file(self):
        """Load images and labels from a text file with external label map."""
        assert os.path.exists(self.image_root_dir), "Invalid image root directory"
        assert self.image_info_file and os.path.exists(self.image_info_file), "Invalid or missing image_info_file"
        """
        image_info_file like this:
        xxx0.png,dog
        xxx1.png,cat
        ...
        """

        """
        label2index_json like this:
        {
	        "dog": 0,
	        "cat": 1,
	        ...
        }
        """
        tmp = []
        with open(self.image_info_file, 'r', encoding='utf-8') as file:
            for line in file:
                image_name, label_name = line.strip().split(',')
                tmp.append((image_name, label_name))

        if self.label2index_json is None:
            tmp_label = list({label for _, label in tmp})
            tmp_label.sort()
            self.label_to_index = dict(zip(tmp_label, range(len(tmp_label))))
        else:
            assert os.path.exists(self.label2index_json), "Invalid label2index"
            with open(self.label2index_json, "r") as f:
                self.label_to_index = json.load(f)

        for image_name, label_name in tmp:
            label_index = self.label_to_index.get(label_name, -1)
            if label_index != -1:
                self.image_names.append(image_name)
                self.image_labels.append(label_index)

    def _load_from_folder_structure(self):
        """Automatically load images and generate label mapping from subfolder names."""
        assert os.path.exists(self.image_root_dir), "Invalid image root directory"

        class_names = sorted([
            name for name in os.listdir(self.image_root_dir)
            if os.path.isdir(os.path.join(self.image_root_dir, name))
        ])

        if self.label2index_json is None:
            self.label_to_index = {name: idx for idx, name in enumerate(class_names)}
        else:
            with open(self.label2index_json, "r") as f:
                self.label_to_index = json.load(f)

        for class_name in class_names:
            class_dir = os.path.join(self.image_root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    relative_path = os.path.join(class_name, filename)
                    self.image_names.append(relative_path)
                    self.image_labels.append(self.label_to_index[class_name])

    def __len__(self):
        return len(self.image_names)

    def get_labels(self):
        """Returns the list of labels (for the full dataset)."""
        return self.image_labels

    def __getitem__(self, index):
        """Load and return image, label, and image name."""
        if self.mode == "text":
            image_path = os.path.join(
                self.image_root_dir,
                self.image_names[index]
            )
        else:  # folder mode
            image_path = os.path.join(self.image_root_dir, self.image_names[index])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, self.image_labels[index], self.image_names[index]
        except (IOError, UnidentifiedImageError) as e:
            print(f"[Warning] Failed to load image {image_path}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Text mode usage
    dataset_from_text = ImageClassificationDataset(
        image_root_dir="../../simple_data",
        image_info_file="../../simple_data/data_info.txt",
        label2index_json="../../simple_data/label2index.json",
        mode="text"
    )
    print(dataset_from_text.index_to_label)
    # Folder mode usage
    dataset_from_folder = ImageClassificationDataset(
        image_root_dir="../../simple_data",
        mode="folder"
    )
    print(dataset_from_folder.index_to_label)
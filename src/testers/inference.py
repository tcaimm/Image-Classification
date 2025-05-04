import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from ..model.Basic_model import load_model
from ..utils.option_parameters import parse_args
from ..data.MeDataset import ImageClassificationDataset
from ..data.image_transform import val_transform
from ..utils.logger_utils import configure_logging

import multiprocessing
def is_main_process() -> bool:
    return multiprocessing.current_process().name == "MainProcess"

if is_main_process():
    logger, timestamp = configure_logging()

def setup_device():
    """
    Set up training device: CUDA if available, else CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

def get_dataloaders(args):
    """
    Create test DataLoaders.
    Handles two modes: 'text' (info file listing) or 'folder' (folder structure).
    Returns loaders and dataset sizes.
    """
    # Define data transforms
    test_tf = val_transform(args.mean, args.std)

    # Initialize datasets
    if args.data_mode == 'text':
        test_dataset = ImageClassificationDataset(
            args.store_all_image_root_path,
            args.test_data_info,
            args.json_file,
            transform=test_tf,
            mode=args.data_mode,
        )
    elif args.data_mode == 'folder':
        test_dataset = ImageClassificationDataset(
            args.test_data_dir,
            transform=test_tf,
            mode=args.data_mode,
        )
    else:
        raise ValueError("data_mode must be either 'text' or 'folder'.")

    # Compute dataset sizes
    num_test = len(test_dataset)
    logger.info(f"Num test examples = {num_test}")
    class_names = [test_dataset.index_to_label[index] for index in range(args.num_classes)]
    logger.info(f"class_names = {class_names}")

    # Create DataLoaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return test_loader, num_test, class_names

def validate(model: torch.nn.Module,
             loader: DataLoader,
             device: torch.device) -> (np.ndarray, np.ndarray, float):
    """
    Run inference on test set and compute accuracy.
    Returns predicted labels, ground truth labels, and accuracy.
    """
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, name in tqdm(loader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    accuracy = 100.0 * (y_pred == y_true).sum() / len(y_true)
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    return y_pred, y_true, accuracy

def analyze_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: list,
                             output_path: str):
    """
    Compute and plot confusion matrix and print classification report.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion Matrix:\n%s", cm)
    report = classification_report(y_true, y_pred, target_names=class_names)
    logger.info("Classification Report:\n" + report)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Annotate cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    logger.info(f"Confusion matrix saved to: {output_path}")
    plt.close(fig)


def main():
    logger.info("Starting Test inference")
    args = parse_args()
    device = setup_device()

    # Load model
    model = load_model(args.model_path, args.model_name, args.num_classes).to(device)

    # Prepare data loader and class names
    test_loader, num_test, class_names = get_dataloaders(args)

    # Run validation
    y_pred, y_true, acc = validate(model, test_loader, device)

    # Analyze confusion matrix
    cm_output = os.path.join(args.test_inference_dir, f"confusion_matrix_{timestamp}.png")
    analyze_confusion_matrix(y_true, y_pred, class_names, cm_output)

    logger.info("Testing complete.")
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers.optimization import get_scheduler

from ..utils.option_parameters import parse_args
from ..data.MeDataset import ImageClassificationDataset
from ..data.image_transform import train_transform, val_transform
from ..model.Basic_model import BasicModel
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
    Create train and validation DataLoaders.
    Handles two modes: 'text' (info file listing) or 'folder' (folder structure).
    Returns loaders and dataset sizes.
    """
    # Define data transforms
    train_tf = train_transform(args.mean, args.std)
    val_tf = val_transform(args.mean, args.std)

    # Initialize datasets
    if args.data_mode == 'text':
        train_dataset = ImageClassificationDataset(
            args.store_all_image_root_path,
            args.train_data_info,
            args.json_file,
            transform=train_tf,
            mode=args.data_mode,
        )
        val_dataset = ImageClassificationDataset(
            args.store_all_image_root_path,
            args.val_data_info,
            args.json_file,
            transform=val_tf,
            mode=args.data_mode,
        )
    elif args.data_mode == 'folder':
        train_dataset = ImageClassificationDataset(
            args.train_data_dir,
            transform=train_tf,
            mode=args.data_mode,
        )
        val_dataset = ImageClassificationDataset(
            args.val_data_dir,
            transform=val_tf,
            mode=args.data_mode,
        )
    else:
        raise ValueError("data_mode must be either 'text' or 'folder'.")

    # Compute dataset sizes
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    logger.info(f"Num train examples = {num_train}")
    logger.info(f"Num val examples = {num_val}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False  # ensure last small batch is used
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, num_train, num_val


def build_model_and_optimizer(args, device, num_train):
    """
    Initialize model, loss, optimizer, and LR scheduler.
    Calculates correct total training steps with gradient accumulation.
    """
    # Model and loss
    model = BasicModel(args.model_name, args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps)

    # Optimizer: keep LR unchanged, handle accumulation in loop
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    steps = math.ceil(num_train / args.batch_size)
    true_steps = math.ceil(steps / args.gradient_accumulation_steps)
    total_steps = args.num_epochs * true_steps

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_steps
    )

    return model, criterion, optimizer, lr_scheduler


def train_one_epoch(
    model, train_loader, criterion, optimizer, scheduler,
    device, epoch, total_epochs, accumulation_steps
):
    """
    Train model for one epoch with gradient accumulation.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        leave=False
    )

    # Zero gradients before starting
    optimizer.zero_grad()

    for step, (inputs, labels, _) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # Normalize loss by accumulation steps
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()

        # Perform optimizer step and lr update every accumulation_steps
        is_last_step = (step + 1) == len(train_loader)
        if (step + 1) % accumulation_steps == 0 or is_last_step:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Metrics
        total_loss += loss.item() * accumulation_steps  # scale back
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f"{total_loss / (step+1):.4f}",
            'acc': f"{100 * correct / total:.2f}%",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * correct / total

    logger.info(
        f"Train Epoch [{epoch}/{total_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%"
    )
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device, epoch, total_epochs):
    """
    Evaluate model on validation set.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} [Val]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    avg_acc = 100.0 * correct / total
    logger.info(
        f"Val   Epoch [{epoch}/{total_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%"
    )
    return avg_loss, avg_acc


def save_model(model, save_dir, epoch, accuracy):
    """
    Save model checkpoint into a folder named by current accuracy.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"epoch_{epoch}_acc_{accuracy:.2f}.pth")
    torch.save(model.state_dict(), path)
    logger.info(f"Saved model to {path}")

def main():
    logger.info("Starting training...")
    # Parse CLI arguments
    args = parse_args()

    # Set up device, data loaders, and model components
    device = setup_device()
    train_loader, val_loader, num_train, num_val = get_dataloaders(args)
    model, criterion, optimizer, scheduler = build_model_and_optimizer(args, device, num_train)

    best_acc = 0.0
    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch,
                                                args.num_epochs, args.gradient_accumulation_steps)

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            args.num_epochs
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, args.save_model_path + "/" + timestamp, epoch, best_acc)
    logger.info("End of training")

if __name__ == '__main__':
    main()
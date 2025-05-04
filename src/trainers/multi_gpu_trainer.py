import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
from contextlib import nullcontext

from ..utils.option_parameters import parse_args
from ..data.MeDataset import ImageClassificationDataset
from ..data.image_transform import train_transform, val_transform
from ..model.Basic_model import BasicModel
from ..utils.logger_utils import configure_logging
from ..data.DataSampler import MyDistributedSampler

from torch.utils.data import DataLoader, DistributedSampler
from diffusers.optimization import get_scheduler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import multiprocessing

logger, timestamp = None, None

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def is_main_process():
    """Returns True if the current process is the main process."""
    return dist.get_rank() == 0 and multiprocessing.current_process().name == "MainProcess"

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



    # Balance Samples
    train_sampler = MyDistributedSampler(train_dataset)

    # train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    # Compute dataset sizes
    num_train = train_sampler.total_size
    num_val = val_sampler.total_size
    if is_main_process():
        logger.info(f"Num train examples = {num_train}")
        logger.info(f"Num val examples = {num_val}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False, # Sampler must with False
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False  # ensure last small batch is used
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
        pin_memory=True
    )

    return train_loader, val_loader, num_train, num_val

def build_model_and_optimizer(args, gpu_id, num_train):
    """
    Initialize model, loss, optimizer, and LR scheduler.
    Calculates correct total training steps with gradient accumulation.
    """
    # Model and loss
    if args.syncBN:
        model = BasicModel(args.model_name, args.num_classes)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(gpu_id)
    else:
        model = BasicModel(args.model_name, args.num_classes).to(gpu_id)

    model = DDP(model, device_ids=[gpu_id])
    criterion = nn.CrossEntropyLoss()

    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * world_size)
    # Optimizer: keep LR unchanged, handle accumulation in loop
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )


    num_samples_per_gpu = math.ceil(num_train / world_size)
    steps_per_epoch = math.ceil(num_samples_per_gpu / args.batch_size)
    true_steps = math.ceil(steps_per_epoch / args.gradient_accumulation_steps)
    total_steps = args.num_epochs * true_steps

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_steps
    )

    return model, criterion, optimizer, lr_scheduler

def reduce_tensor(tensor, average=True):
    """
    Reduces a tensor across all processes.
    """
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= dist.get_world_size()
    return tensor

def train_one_epoch(
    model, train_loader, criterion, optimizer, scheduler,
    gpu_id, epoch, total_epochs, accumulation_steps
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    if is_main_process():
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    else:
        pbar = train_loader

    optimizer.zero_grad()

    for step, (inputs, labels, _) in enumerate(pbar):
        inputs, labels = inputs.to(gpu_id), labels.to(gpu_id)
        is_sync_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(train_loader))

        context = model.no_sync if not is_sync_step else nullcontext
        with context():
            outputs = model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()

        if is_sync_step:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if is_main_process():
            pbar.set_postfix({
                'loss': f"{total_loss / (step + 1):.4f}",
                'acc': f"{100 * correct / total:.2f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })

    total_loss_tensor = torch.tensor(total_loss, device=gpu_id)
    correct_tensor = torch.tensor(correct, device=gpu_id)
    total_tensor = torch.tensor(total, device=gpu_id)

    total_loss = reduce_tensor(total_loss_tensor).item()
    correct = reduce_tensor(correct_tensor, average=False).item()
    total = reduce_tensor(total_tensor, average=False).item()

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * correct / total

    if is_main_process():
        logger.info(
            f"Train Epoch [{epoch}/{total_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%"
        )

    return avg_loss, avg_acc

def validate(model, val_loader, criterion, gpu_id, epoch, total_epochs):
    """
    Evaluate model on validation set.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    if is_main_process():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} [Val]", leave=False)
    else:
        pbar = val_loader

    with torch.no_grad():
        for inputs, labels, _ in pbar:
            inputs, labels = inputs.to(gpu_id), labels.to(gpu_id)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    total_loss_tensor = torch.tensor(total_loss, device=gpu_id)
    correct_tensor = torch.tensor(correct, device=gpu_id)
    total_tensor = torch.tensor(total, device=gpu_id)

    total_loss = reduce_tensor(total_loss_tensor).item()
    correct = reduce_tensor(correct_tensor, average=False).item()
    total = reduce_tensor(total_tensor, average=False).item()

    avg_loss = total_loss / len(val_loader)
    avg_acc = 100.0 * correct / total

    if is_main_process():
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
    if isinstance(model, DDP):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    if is_main_process():
        logger.info(f"Saved model to {path}")

def main():
    ddp_setup()

    if is_main_process():
        global logger, timestamp
        logger, timestamp = configure_logging()
        logger.info("Starting Training ...")

    args = parse_args()
    gpu_id = int(os.environ["LOCAL_RANK"])

    train_loader, val_loader, num_train, num_val = get_dataloaders(args)
    model, criterion, optimizer, scheduler = build_model_and_optimizer(args, gpu_id, num_train)

    best_acc = 0.0
    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, gpu_id, epoch,
                                                args.num_epochs, args.gradient_accumulation_steps)

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            gpu_id,
            epoch,
            args.num_epochs
        )

        # Save best model
        if is_main_process() and val_acc > best_acc:
            best_acc = val_acc
            save_model(model, args.save_model_path + "/" + timestamp, epoch, best_acc)
    if is_main_process():
        logger.info("End Training!")
    if dist.is_initialized():
        destroy_process_group()
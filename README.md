# Image Classification

This project provides a flexible PyTorch-based implementation for image classification tasks. It supports multiple model architectures (ResNet-34, Swin Transformer V2-B) and can be easily extended to include additional models.

> **Note:** This project has been tested and verified only on **Ubuntu** systems. Compatibility with Windows or macOS is not guaranteed at this stage.

---

## Features

- Multiple pre-trained model support (ResNet34, Swin_v2_b ...)
- Flexible dataset loading from folder structure or text file definitions
- Multi-GPU training support (single machine with multiple cards)
- Sample Balance Strategy Distributed Upsampling
- Gradient accumulation for effective batch size control
- Learning rate scheduler with warmup
- Comprehensive logging and model checkpointing
-

## Installation

### 1. Install PyTorch

Please install the appropriate version of PyTorch for your system (OS, CUDA version, etc.) by following the instructions on the official PyTorch website: https://pytorch.org/get-started/locally

Example (for CUDA 12.4):
```bash
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu124
```

### 2. Clone the repository and install the required dependencies:
```bash
git clone https://github.com/tcaimm/Image-Classification.git
cd Image-Classification
pip install -r requirements.txt
```

## Dataset Preparation

The project supports two data loading modes:

### 1. Text Mode
Place all your image files in a single folder. The directory structure should look like this:
```
store_all_image_root_path/
├── image1.png
├── image2.png
├── image3.png
├── image4.png
└── ... (more images)
```

Create a data info text file with image names and labels:
```
image1.png,class1
image2.png,class2
...
```

Create a JSON file mapping class names to indices:
```json
{
    "class1": 0,
    "class2": 1,
    ...
}
```
This JSON file can also be omitted

### 2. Folder Mode

Organize your dataset in the following structure:
```
dataset_root/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image3.jpg
│   ├── image4.jpg
│   └── ...
└── ...
```

## Training

Use the provided training scripts single GPU:

```bash
cd scripts
bash train.sh
```

Or for multi-GPU training:

```bash
cd scripts
bash train_multi.sh
```

You can customize training parameters in the script:

- Model architecture (`--model_name`)
- Number of classes (`--num_classes`)
- Batch size (`--batch_size`)
- Learning rate (`--learning_rate`)
- Number of epochs (`--num_epochs`)
- Learning rate warmup steps(`--lr_warmup_steps`) TotalSteps * (3%–10%) \
TotalSteps = ceil(N / (BatchSize × WorldSize)) × (NumEpochs / AccumulationSteps)
- Data loading mode (`--data_mode`)
- ......(other parameters in src/utils/option_parameters.py)

## Inference

Use the test scripts for inference:

```bash
cd scripts
bash test.sh
```

Or for multi-GPU testing:

```bash
cd scripts
bash test_multi.sh
```

## Tools

The project includes several utility tools:

- `tools/split_dataset.py`: Split your dataset into training and validation sets
- `tools/calculate_mean_std.py`: Calculate the mean and standard deviation of your dataset for normalization

## Project Structure

```
.
├── scripts/                # Training and inference scripts
│   ├── train.sh            # Single-GPU training
│   ├── train_multi.sh      # Multi-GPU training
│   ├── test.sh             # Single-GPU inference
│   └── test_multi.sh       # Multi-GPU inference
├── src/
│   ├── data/               # Dataset and data loading
│   ├── model/              # Model architectures
│   ├── trainers/           # Training implementations
│   │   ├── trainer.py      # Single-GPU trainer
│   │   └── multi_gpu_trainer.py # Multi-GPU trainer
│   ├── testers/            # Testing implementations
│   └── utils/              # Utility functions
├── tools/                  # Data preparation tools
└── simple_data/            # Example data
```

## Examples

### Training Example

```bash
cd scripts
export CUDA_VISIBLE_DEVICES=0
python train_and_val.py \
    --data_mode "text" \
    --store_all_image_root_path "/your_dataset/all_data" \
    --train_data_info "/your_dataset/data_info.txt" \
    --val_data_info "/your_dataset/data_info.txt" \
    --json_file "/your_dataset/label2index.json" \
    --model_name "resnet34" \
    --num_classes 3 \
    --batch_size 32 \
    --num_epochs 20
```

### Inference Example

```bash
cd scripts
export CUDA_VISIBLE_DEVICES=0
python test_inference.py \
    --data_mode "text" \
    --store_all_image_root_path "/your_dataset/all_data" \
    --test_data_info "/your_dataset/data_info.txt" \
    --json_file "/your_dataset/label2index.json" \
    --model_name "resnet34" \
    --model_path "./saved_model/latest/model.pth" \
    --num_classes 3
```

## License

This project is licensed under the [MIT License](LICENSE).

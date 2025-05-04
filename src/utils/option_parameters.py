import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch implementation of image classification.")
    parser.add_argument("--data_mode", type=str, required=True, help="mode must in ('text', 'folder')")
    parser.add_argument("--store_all_image_root_path", type=str, help="store the image path.")
    parser.add_argument("--train_data_info", type=str, help="train data info.")
    parser.add_argument("--val_data_info", type=str, help="val data info.")
    parser.add_argument("--json_file", type=str, help="Path to the label2code directory.")

    parser.add_argument("--train_data_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("--val_data_dir", type=str, help="Path to the dataset directory.")

    parser.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406], help="Mean for normalization (RGB)")
    parser.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225], help="Std for normalization (RGB)")

    parser.add_argument("--model_name", type=str, required=True, help="store the image path.")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes in the dataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="gradient accumulation steps.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type=int, default=30, help="Number of epochs for the warmup in the lr scheduler.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 penalty) for the optimizer.")
    parser.add_argument("--save_model_path", type=str, default="./saved_model", help="Directory to save the trained model.")

    # test inference
    parser.add_argument("--test_data_info", type=str, help="test data info.")
    parser.add_argument("--test_data_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("--model_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("--test_inference_dir", type=str, default="./test_info", help="Directory to save the inference info.")

    return parser.parse_args()
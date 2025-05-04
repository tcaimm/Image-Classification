from torchvision import transforms

def train_transform(mean, std, size=224):
    return transforms.Compose([
        transforms.Resize((int(size * 1.05), int(size * 1.05))),  # Slightly enlarge the image
        transforms.RandomCrop(size),                              # Random crop to the target size
        transforms.RandomHorizontalFlip(),                       # Random horizontal flip
        transforms.RandomVerticalFlip(),                         # Random vertical flip
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Color augmentation
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),                   # Add Gaussian blur
        transforms.ToTensor(),                                   # Convert to tensor
        transforms.Normalize(mean=mean, std=std),                # Normalize using mean and std
        transforms.RandomErasing(value="random", scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Random erasing (small areas)
    ])

def val_transform(mean, std, size=256):
    return transforms.Compose([
        transforms.Resize((size, size)),     # Resize to target size
        transforms.CenterCrop(size),         # Center crop
        transforms.ToTensor(),               # Convert to tensor
        transforms.Normalize(mean=mean, std=std),  # Normalize using mean and std
    ])

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights, swin_v2_b, Swin_V2_B_Weights

class BasicModel(nn.Module):
    """
    A flexible model class that supports different architectures and allows
    modification of the final classification layer to accommodate a specified
    number of output classes.
    """

    def __init__(self, model_name='resnet34', num_classes=3, seed=42):
        """
        Initializes the BasicModel with a specified architecture and number of output classes.

        Args:
            model_name (str): Name of the model architecture ('resnet34' or 'swin_v2_b').
            num_classes (int): Number of output classes for the classification task.
            seed (int): Random seed for weight initialization consistency.
        """
        super(BasicModel, self).__init__()
        torch.manual_seed(seed)  # Set random seed for reproducibility

        # Initialize the specified model architecture
        if model_name == 'resnet34':
            # Load pre-trained ResNet-34 model with ImageNet weights
            self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            # Get the number of input features of the final fully connected layer
            num_features = self.model.fc.in_features
            # Replace the final fully connected layer with a new one for the specified number of classes
            self.model.fc = nn.Linear(num_features, num_classes)

        elif model_name == 'swin_v2_b':
            self.model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
            num_features = self.model.head.in_features
            self.model.head = nn.Linear(num_features, num_classes)

        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Choose 'resnet34' or 'swin_v2_b'.")

    def forward(self, x):
        return self.model(x)

def load_model(model_path, model_name='resnet34', num_classes=3):
    """
    Loads a trained model from a specified file path.

    Args:
        model_path (str): Path to the saved model state dictionary.
        model_name (str): Name of the model architecture ('resnet34' or 'swin_v2_b').
        num_classes (int): Number of output classes for the classification task.

    Returns:
        BasicModel: The loaded model in evaluation mode.
    """
    # Initialize the model with the specified architecture and number of classes
    model = BasicModel(model_name=model_name, num_classes=num_classes)
    # Load the state dictionary from the specified file
    state_dict = torch.load(model_path, map_location='cpu')
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] if k.startswith('module.') else k  # 移除 'module.'
    #     new_state_dict[name] = v

    model.eval()
    return model

if __name__ == "__main__":
    # Example usage
    model_name = 'resnet34'  # Choose between 'resnet34' and 'swin_v2_b'
    num_classes = 3  # Define the number of output classes
    model = BasicModel(model_name=model_name, num_classes=num_classes)
    print(model)

    # Create a random input tensor with the appropriate size
    if model_name == 'resnet34':
        x = torch.randn((2, 3, 224, 224))  # ResNet-34 expects 224x224 images
    elif model_name == 'swin_v2_b':
        x = torch.randn((2, 3, 256, 256))  # Swin V2 B expects at least 256x256 images

    # Perform a forward pass through the model
    y = model(x)
    # Apply softmax to obtain probabilities
    probabilities = torch.softmax(y, dim=1)
    # Get the predicted class with the highest probability
    max_probs, preds = torch.max(probabilities, 1)
    print(f"Predicted classes: {preds}")
    print(f"Output shape: {y.shape}")

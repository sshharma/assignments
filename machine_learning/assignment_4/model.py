# model.py
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model():
    # Load the pretrained ResNet50 model with the default weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Uncomment the below lines if you wish to freeze the model's layers, considering the smaller dataset
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the final layer to output 2 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model

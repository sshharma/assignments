"""
Name: Sachin Sharma
KSUID: 001145317
Project: 3
Title: Deep Learning for Classification
"""


import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model():
    # Load the pretrained ResNet50 model with the default weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Uncomment the below lines if you wish to freeze the model's layers, considering the smaller dataset
    # for param in model.parameters():                  # results of this run can be find in results.csv file
    #     param.requires_grad = False

    # Modify the final layer to output 2 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model

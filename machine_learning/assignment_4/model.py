# model.py
import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    # Modify the final layer to output 2 classes
    model.fc = nn.Linear(num_ftrs, 2)
    return model

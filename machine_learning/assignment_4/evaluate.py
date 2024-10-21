"""
Name: Sachin Sharma
KSUID: 001145317
Project: 3
Title: Deep Learning for Classification
"""


import torch
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

def evaluate_model(model, dataloader, device):
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in dataloader['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    # Compute metrics
    cm = confusion_matrix(labels_list, preds_list)
    accuracy = accuracy_score(labels_list, preds_list)
    sensitivity = recall_score(labels_list, preds_list)
    specificity = recall_score(labels_list, preds_list, pos_label=0)

    return cm, accuracy, sensitivity, specificity

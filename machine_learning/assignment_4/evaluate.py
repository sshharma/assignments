"""
Name: Sachin Sharma
KSUID: 001145317
Project: 3
Title: Deep Learning for Classification
"""


import torch
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve, auc
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device):
    model.eval()
    preds_list, probs_list = [], []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in dataloader['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)   # Get probabilities using softmax (assumes binary classification)
            positive_probs = probs[:, 1]            # Extract probability for the positive class (index 1)
            probs_list.extend(positive_probs.cpu().numpy())

            _, preds = torch.max(outputs, 1)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    # Compute metrics
    cm = confusion_matrix(labels_list, preds_list)
    accuracy = accuracy_score(labels_list, preds_list)
    sensitivity = recall_score(labels_list, preds_list)
    specificity = recall_score(labels_list, preds_list, pos_label=0)
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels_list, probs_list)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return cm, accuracy, sensitivity, specificity

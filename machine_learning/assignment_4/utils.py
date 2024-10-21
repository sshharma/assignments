"""
Name: Sachin Sharma
KSUID: 001145317
Project: 3
Title: Deep Learning for Classification
"""

import csv
import os
from filelock import FileLock

def save_results(csv_file, params, metrics):
    lock_file = csv_file + '.lock'
    with FileLock(lock_file):
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as csvfile:
            headers = [
                'Learning Rate', 'Batch Size', 'Max Epochs',
                'Accuracy', 'Sensitivity', 'Specificity',
                'CM_TN', 'CM_FP', 'CM_FN', 'CM_TP',
                'Model Path'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=headers)

            if not file_exists:
                writer.writeheader()

            cm = metrics['confusion_matrix']
            row = {
                'Learning Rate': params['learning_rate'],
                'Batch Size': params['batch_size'],
                'Max Epochs': params['max_epochs'],
                'Accuracy': metrics['accuracy'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'CM_TN': cm[0][0],
                'CM_FP': cm[0][1],
                'CM_FN': cm[1][0],
                'CM_TP': cm[1][1],
                'Model Path': metrics['model_path']
            }
            writer.writerow(row)

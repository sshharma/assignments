"""
Name: Sachin Sharma
KSUID: 001145317
Project: 4
Title: Deep Learning for Classification
"""



import argparse
import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import RetinaDataset
from model import get_model
from train import train_model
from evaluate import evaluate_model
from utils import save_results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='DR vs NonDR Classification')
    parser.add_argument('--learning-rate', type=float, default= 0.0005 , help='Learning rate')
    parser.add_argument('--batch-size', type=int, default= 32 , help='Minibatch size')
    parser.add_argument('--max-epochs', type=int, default= 10, help='Number of epochs')
    parser.add_argument('--data-dir', type=str, default='DS_IDRID/', help='Path to the dataset directory')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--freeze-layers', action= 'store_true' ,help='Freeze layers')
    args = parser.parse_args()

    # Parameters
    params = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs
    }

    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    # device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    train_dir = os.path.join(args.data_dir, 'Train')
    test_dir = os.path.join(args.data_dir, 'Test')

    # Datasets and dataloaders
    datasets = {
        'train': RetinaDataset(root_dir=train_dir, transform=data_transforms['train']),
        'test': RetinaDataset(root_dir=test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=params['batch_size'], shuffle=True),
        'test': DataLoader(datasets['test'], batch_size=params['batch_size'], shuffle=False)
    }

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, criterion, optimizer
    model = get_model(freeze= args.freeze_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, params['max_epochs'], device)

    # Generate a unique filename for the model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"resnet_lr{params['learning_rate']}_bs{params['batch_size']}_epochs{params['max_epochs']}_{timestamp}.pth"
    model_filepath = os.path.join('saved_models', model_filename)

    # Create directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)

    # Save the model
    # torch.save(model.state_dict(), model_filepath)

    # Evaluate the model
    cm, accuracy, sensitivity, specificity = evaluate_model(model, dataloaders, device)

    # Print confusion matrix and metrics
    print('Confusion Matrix:')
    print(cm)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Sensitivity (Recall for class 1): {sensitivity:.4f}')
    print(f'Specificity (Recall for class 0): {specificity:.4f}')

    # Save results
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'model_path': model_filepath
    }
    save_results('results.csv', params, metrics)

if __name__ == '__main__':
    main()

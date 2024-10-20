# data_loader.py
import os
from torch.utils.data import Dataset
from PIL import Image

class RetinaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.jpg'):
                label = self._get_label(filename)
                if label is not None:
                    self.image_paths.append(os.path.join(self.root_dir, filename))
                    self.labels.append(label)

    def _get_label(self, filename):
        if filename.endswith('-0.jpg'):
            return 0  # NonDR
        elif filename.endswith('-3.jpg') or filename.endswith('-4.jpg'):
            return 1  # DR
        else:
            return None  # Discard labels 1 and 2

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

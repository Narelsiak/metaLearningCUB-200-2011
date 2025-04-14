import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class BirdDataset(data.Dataset):
    """
    Comprehensive dataset class for bird image classification.
    Handles data loading, preprocessing, splitting and transformations.
    """
    
    def __init__(self, dataframe, image_folder, image_size=(224, 224), mode='train', num_workers=0):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'image_name' and 'class_id' columns
            image_folder (str): Path to folder containing images
            image_size (tuple): Target image size (width, height)
            mode (str): One of 'train', 'val', or 'test' to specify dataset type
        """
        super().__init__()
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.image_size = image_size
        self.mode = mode
        self.num_workers=num_workers
        
        # Precomputed normalization statistics for the dataset
        self.data_means = torch.Tensor([0.5183975, 0.49192241, 0.44651328])
        self.data_std = torch.Tensor([0.26770132, 0.25828985, 0.27961241])
        
        # Set random seed for reproducibility
        torch.manual_seed(0)
        
        # Split data if needed
        if 'split' not in self.dataframe.columns:
            self._split_data()
        
        # Filter data based on mode
        self.df = self.dataframe[self.dataframe['split'] == mode]
        
        # Load images and labels
        self.images, self.labels = self._load_images(self.df)
        
        # Get appropriate transforms
        self.transform = self._get_transforms()

    def _split_data(self):
        """Internal method to split data into training, validation and test sets."""
        unique_classes = self.dataframe['class_id'].unique()
        randomized_classes = torch.randperm(len(unique_classes))
        
        # Split classes: 160 training, 20 validation, 20 test
        train_classes = unique_classes[randomized_classes[:160]]
        val_classes = unique_classes[randomized_classes[160:180]]
        test_classes = unique_classes[randomized_classes[180:]]
        
        # Assign split labels
        self.dataframe['split'] = 'train'
        self.dataframe.loc[self.dataframe['class_id'].isin(val_classes), 'split'] = 'val'
        self.dataframe.loc[self.dataframe['class_id'].isin(test_classes), 'split'] = 'test'

    @staticmethod
    def _load_single_image(row, image_folder, image_size):
        img_path = os.path.join(image_folder, row['image_name'])
        img = Image.open(img_path).convert('RGB').resize(image_size)
        img_array = np.array(img)
        return img_array, row['class_id']

    def _load_images(self, dataframe):
        images = []
        labels = []

        func = partial(self._load_single_image, image_folder=self.image_folder, image_size=self.image_size)
        rows = [row for _, row in dataframe.iterrows()]
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(func, rows))

        for img_array, label in results:
            images.append(img_array)
            labels.append(label)

        return np.array(images), torch.tensor(labels)

    def _get_transforms(self):
        """Internal method to get appropriate transforms based on dataset mode."""
        if self.mode == 'train':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(self.data_means, self.data_std)
            ])
        else:  # val or test
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(self.data_means, self.data_std)
            ])

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.images)

    @classmethod
    def create_datasets(cls, dataframe, image_folder, image_size=(224, 224), num_workers=0):
        """
        Class method to create all three datasets (train, val, test) at once.
        
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        # First ensure data is split
        if 'split' not in dataframe.columns:
            temp_ds = cls(dataframe, image_folder, image_size, mode='train', num_workers=num_workers)
            dataframe = temp_ds.dataframe
        
        train_ds = cls(dataframe, image_folder, image_size, mode='train', num_workers=num_workers)
        val_ds = cls(dataframe, image_folder, image_size, mode='val', num_workers=num_workers)
        test_ds = cls(dataframe, image_folder, image_size, mode='test', num_workers=num_workers)
        
        return train_ds, val_ds, test_ds
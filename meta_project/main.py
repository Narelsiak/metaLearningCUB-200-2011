from data.data_loader import DataLoader
from data.bird_data import BirdDataset
from data.custom_batch_sampler import CustomBatchSampler
from utils.split_batch import split_batch

from figures.data_grid import show_fewshot_batch
import os
import torch.utils.data as data

import torchvision
IMAGE_FOLDER = os.getcwd() + "/data/raw/CUB_200_2011/images/"
N_WAY = 5
K_SHOT = 4

if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.load_and_merge_data()

    train_dataset, val_dataset, test_dataset = BirdDataset.create_datasets(df, IMAGE_FOLDER)

    print(f"Train labels count: {len(set(train_dataset.labels))}")
    print(f"Val labels count: {len(set(val_dataset.labels))}")

    train_sampler = CustomBatchSampler(
        train_dataset.labels,
        N_way=N_WAY,
        K_shot=K_SHOT,
        support_query_mode =True
    )
    val_sampler = CustomBatchSampler(
        val_dataset.labels,
        N_way=N_WAY,
        K_shot=K_SHOT,
        support_query_mode =True,
    )

    print(f"Train sampler length: {len(train_sampler)}")
    print(f"Val sampler length: {len(val_sampler)}")

    train_data_loader = data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_data_loader = data.DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)

    imgs, targets = next(iter(val_data_loader))
    show_fewshot_batch(imgs, targets, k_shot=K_SHOT)
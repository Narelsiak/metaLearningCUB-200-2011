from data.data_loader import DataLoader
from data.bird_data import BirdDataset
from data.custom_batch_sampler import CustomBatchSampler
from data.MAML_batch_sampler import TaskBatchSampler

import os
import torch
import torch.utils.data as data

from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from meta_project.modeling.ProtoMAML import ProtoMAML

IMAGE_FOLDER = os.getcwd() + "/data/raw/CUB_200_2011/images/"
N_WAY = 5
K_SHOT = 4
CHECKPOINT = 'models/'
model_path = 'models/model'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_model(model_class, train_loader, val_loader, **kwargs):
    # Prepare logging
    log_dir = os.path.join(CHECKPOINT, model_class.__name__, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(CHECKPOINT, model_class.__name__),
        name="tensorboard_logs",
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT, model_class.__name__),
        filename="best-{epoch}-{val_acc:.2f}",
        save_weights_only=True,
        mode="max",
        monitor="val_acc",
        save_top_k=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Trainer configuration
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT, model_class.__name__),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=200,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=True
    )
    
    pl.seed_everything(42)  # For reproducibility
    model = model_class(**kwargs)
    
    # Train with validation
    trainer.fit(model, train_loader, val_loader)
    
    print("Best model saved at:", trainer.checkpoint_callback.best_model_path)


    # Load best model after training
    model = model_class.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    
    # Save the best model for future use
    trainer.save_checkpoint(model_path)
    
    return model

if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.load_and_merge_data()

    train_dataset, val_dataset, test_dataset = BirdDataset.create_datasets(df, IMAGE_FOLDER, num_workers=10)

    print(f"Train labels count: {len(set(train_dataset.labels))}")
    print(f"Val labels count: {len(set(val_dataset.labels))}")

    train_maml_sampler = TaskBatchSampler(train_dataset.labels, include_query=True, N_way=N_WAY, K_shot=K_SHOT, batch_size=16)
    train_maml_loader = data.DataLoader(train_dataset, batch_sampler=train_maml_sampler, collate_fn=train_maml_sampler.get_collate_fn()) #""", num_workers=2, persistent_workers=True, pin_memory=True""" 

    val_maml_sampler = TaskBatchSampler(val_dataset.labels, include_query=True, N_way=N_WAY, K_shot=K_SHOT, batch_size=1)
    val_maml_loader = data.DataLoader(val_dataset, batch_sampler=val_maml_sampler, collate_fn=val_maml_sampler.get_collate_fn())


    protomaml_model = train_model(ProtoMAML,
                              proto_dim=64,
                              lr=1e-3,
                              lr_inner=0.1,
                              lr_output=0.1,
                              num_inner_steps=1,  # Often values between 1 and 10
                              train_loader=train_maml_loader,
                              val_loader=val_maml_loader)
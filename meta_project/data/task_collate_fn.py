import torch

def task_collate_fn(batch, task_batch_size):
    imgs = torch.stack([img for img, target in batch], dim=0)
    targets = torch.stack([target for img, target in batch], dim=0)
    imgs = imgs.chunk(task_batch_size, dim=0)
    targets = targets.chunk(task_batch_size, dim=0)
    return list(zip(imgs, targets))
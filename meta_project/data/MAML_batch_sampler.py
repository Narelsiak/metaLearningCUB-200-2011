import torch
from torch.utils.data.dataloader import default_collate
from .custom_batch_sampler import CustomBatchSampler

class TaskBatchSampler(object):
    """
    Adapter that aggregates multiple few-shot tasks into batches of tasks.
    Wraps around CustomBatchSampler to enable meta-batching for MAML-like algorithms.
    
    Args:
        dataset_targets: PyTorch tensor of dataset labels
        batch_size: Number of tasks per meta-batch
        N_way: Number of classes per task
        K_shot: Number of examples per class (will be doubled if include_query=True)
        include_query: If True, each task will have K_shot support and K_shot query examples
        shuffle: Whether to shuffle tasks and examples
    """
    def __init__(self, dataset_targets, batch_size, N_way, K_shot, include_query=False, shuffle=True):
        super().__init__()
        self.task_sampler = CustomBatchSampler(
            dataset_targets=dataset_targets,
            N_way=N_way,
            K_shot=K_shot,
            support_query_mode=include_query,
            shuffle=shuffle
        )
        self.task_batch_size = batch_size  # Number of tasks per meta-batch
        self.samples_per_task = len(self.task_sampler[0])  # Samples per single task

    def __iter__(self):
        """Yields batches containing multiple tasks"""
        task_batch = []
        for task in self.task_sampler:  # Each task is a list of sample indices
            task_batch.extend(task)
            if len(task_batch) == self.task_batch_size * self.samples_per_task:
                yield task_batch
                task_batch = []
        
        # Yield remaining tasks if any
        if len(task_batch) > 0:
            yield task_batch

    def __len__(self):
        """Number of meta-batches (batches of tasks)"""
        return len(self.task_sampler) // self.task_batch_size

    def get_collate_fn(self):
        """Returns collate function that organizes data into list of (imgs, targets) tuples per task"""
        def collate_fn(batch):
            # batch contains all examples from all tasks in the meta-batch
            if isinstance(batch[0], (tuple, list)):
                # Case when batch contains (img, target) pairs
                imgs = torch.stack([item[0] for item in batch], dim=0)
                targets = torch.stack([item[1] for item in batch], dim=0)
                
                # Split into tasks
                imgs = imgs.chunk(self.task_batch_size, dim=0)
                targets = targets.chunk(self.task_batch_size, dim=0)
                
                return list(zip(imgs, targets))
            else:
                # Case when batch contains only imgs (no targets)
                imgs = torch.stack(batch, dim=0)
                return list(imgs.chunk(self.task_batch_size, dim=0))
        
        return collate_fn
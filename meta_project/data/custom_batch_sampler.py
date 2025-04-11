import torch
import random
from collections import defaultdict
import numpy as np

class CustomBatchSampler(object):
    """
    A batch sampler for few-shot learning tasks that generates episodes containing
    N classes with K examples each (support + query when support_query_mode=True).
    
    Args:
        dataset_targets (torch.Tensor): Tensor containing labels for all dataset samples
        N_way (int): Number of classes per episode
        K_shot (int): Number of examples per class
        support_query_mode (bool): If True, generates both support and query sets 
                                 (doubles K_shot internally)
        shuffle (bool): Whether to shuffle classes/examples each iteration (for training)
        shuffle_once (bool): Whether to shuffle once initially (for validation)
    """

    def __init__(self, dataset_targets, N_way, K_shot, support_query_mode=False, 
                 shuffle=True, shuffle_once=False):
        super().__init__()
        self.dataset_targets = dataset_targets
        self.N_way = N_way
        # Double K_shot if we need both support and query samples
        self.K_shot = K_shot * 2 if support_query_mode else K_shot  
        self.shuffle = shuffle
        self.support_query_mode = support_query_mode

        # Organize data by class: {class: [indices]}
        self.classes, self.class_indices = self._organize_by_class(dataset_targets)
        
        # Create list of available class batches
        self.class_list = self._create_class_list()

        # Initial shuffling if needed
        if shuffle_once or shuffle:
            self._shuffle_data()
        else:
            self._sort_for_testing()

    def _organize_by_class(self, targets):
        """Group dataset indices by their class labels.
        Returns:
            classes: List of unique class labels
            class_indices: Dict {class: tensor_of_indices}
        """
        classes = torch.unique(targets).tolist()
        class_indices = {c: torch.where(targets == c)[0] for c in classes}
        return classes, class_indices

    def _create_class_list(self):
        """Create a list of classes where each class appears as many times as 
        the number of K_shot batches it can provide.
        Also calculates total iterations (episodes) possible.
        """
        batches_per_class = {
            c: len(indices) // self.K_shot 
            for c, indices in self.class_indices.items()
        }
        self.iterations = sum(batches_per_class.values()) // self.N_way
        return [c for c in self.classes for _ in range(batches_per_class[c])]

    def _shuffle_data(self):
        """Shuffle both:
        1. The order of examples within each class
        2. The order of classes in class_list
        """
        # Shuffle examples per class
        for c in self.classes:
            perm = torch.randperm(len(self.class_indices[c]))
            self.class_indices[c] = self.class_indices[c][perm]
        # Shuffle class order
        random.shuffle(self.class_list)

    def _sort_for_testing(self):
        """Sort classes in deterministic order for testing/reproducibility"""
        sort_idxs = [
            i + p*len(self.classes) 
            for i, c in enumerate(self.classes) 
            for p in range(len(self.class_indices[c]) // self.K_shot)
        ]
        self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def __iter__(self):
        """Generates batches of indices for few-shot episodes.
        Each episode contains N_way classes with K_shot examples per class.
        When support_query_mode=True, alternates support/query samples.
        """
        if self.shuffle:
            self._shuffle_data()

        start_index = defaultdict(int)  # Tracks next available index per class
        
        for it in range(self.iterations):
            # Select N classes for this episode
            batch_classes = self.class_list[it*self.N_way:(it+1)*self.N_way]
            
            # Get K_shot examples per class
            indices = [self._get_class_indices(c, start_index) for c in batch_classes]
            
            # Flatten into single list
            batch = [idx for class_indices in indices for idx in class_indices]
            
            # Interleave support/query samples if needed
            if self.support_query_mode:
                batch = batch[::2] + batch[1::2]  # [s1, q1, s2, q2, ...]
                
            yield batch

    def _get_class_indices(self, c, start_index):
        """Get next K_shot indices for a class and update start position"""
        end = start_index[c] + self.K_shot
        indices = self.class_indices[c][start_index[c]:end]
        start_index[c] = end
        return indices

    def __len__(self):
        """Returns number of episodes (batches) available"""
        return self.iterations
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.split_batch import split_batch

def show_fewshot_batch(imgs, targets, k_shot=5):
    def denormalize_tensor(tensor, value_range=(-1, 1), scale_each=True):
        tensor = tensor.clone()

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 3:
            if tensor.size(0) == 1:
                tensor = torch.cat((tensor, tensor, tensor), 0)
            tensor = tensor.unsqueeze(0)

        if tensor.dim() == 4 and tensor.size(1) == 1:
            tensor = torch.cat((tensor, tensor, tensor), 1)
            
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

        return tensor
    
    def create_image_grid(images, nrow):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        if images.dtype != np.float32:
            images = images.astype(np.float32) / 255.0

        n_images = images.shape[0]
        nrow = min(nrow, n_images)
        ncol = (n_images + nrow - 1) // nrow
        H, W, C = images.shape[1], images.shape[2], images.shape[3]

        grid = np.ones((ncol * H, nrow * W, C)) * 0.9

        for i in range(n_images):
            row = i // nrow
            col = i % nrow
            grid[row*H:(row+1)*H, col*W:(col+1)*W, :] = images[i]

        return grid

    support_imgs, query_imgs, _, _ = split_batch(imgs, targets)
    
    support_imgs = denormalize_tensor(support_imgs, scale_each=True)
    query_imgs = denormalize_tensor(query_imgs, scale_each=True)

    support_grid = create_image_grid(support_imgs, nrow=k_shot)
    query_grid = create_image_grid(query_imgs, nrow=k_shot)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].imshow(support_grid)
    ax[0].set_title(f"Support Set ({k_shot}-shot)", fontsize=12)
    ax[0].axis('off')

    ax[1].imshow(query_grid)
    ax[1].set_title("Query Set", fontsize=12)
    ax[1].axis('off')

    plt.suptitle("Few-Shot Learning Batch", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.close()

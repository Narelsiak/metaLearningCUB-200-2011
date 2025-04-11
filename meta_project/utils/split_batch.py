def split_batch(imgs, targets):
    split_idx = len(imgs) // 2
    support_imgs, query_imgs = imgs[:split_idx], imgs[split_idx:]
    support_targets, query_targets = targets[:split_idx], targets[split_idx:]
    
    return support_imgs, query_imgs, support_targets, query_targets
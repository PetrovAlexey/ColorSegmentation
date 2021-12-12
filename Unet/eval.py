import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

def multi_acc(pred, true):
    acc = np.count_nonzero(pred.numpy()-true.numpy() == 0) / pred.nelement()
    return acc

def eval_net(net, loader, device):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                probs = F.softmax(mask_pred, dim=1)
                mask_pred = probs.cpu()
                mask_pred = torch.argmax(mask_pred,dim=1).squeeze(0)

            if net.n_classes > 1:
                tot += multi_acc(true_masks.squeeze(1).cpu(), mask_pred.cpu())
            pbar.update()

    net.train()
    return tot / n_val

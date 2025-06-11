import os
import torchvision.utils as vutils
import torch

def save_prediction_images(t1, t2, pred, label, out_dir, idx):
    os.makedirs(out_dir, exist_ok=True)

    # Ensure shape [3, H, W] for all
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if label.dim() == 2:
        label = label.unsqueeze(0)
    if pred.size(0) == 1:
        pred = pred.repeat(3, 1, 1)
    if label.size(0) == 1:
        label = label.repeat(3, 1, 1)

    t1 = t1.float()
    t2 = t2.float()
    if t1.max() > 1.0:
        t1 /= 255.0
    if t2.max() > 1.0:
        t2 /= 255.0

    grid = vutils.make_grid([t1, t2, pred, label], nrow=4)
    vutils.save_image(grid, os.path.join(out_dir, f"sample_{idx}.png"))

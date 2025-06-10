import os
import torchvision.utils as vutils

def save_prediction_images(t1, t2, pred, label, out_dir, idx):
    os.makedirs(out_dir, exist_ok=True)

    # Ensure 3 channels for all
    if t1.shape[0] == 1:
        t1 = t1.repeat(3, 1, 1)
    if t2.shape[0] == 1:
        t2 = t2.repeat(3, 1, 1)
    if pred.shape[0] == 1:
        pred = pred.repeat(3, 1, 1)
    if label.shape[0] == 1:
        label = label.repeat(3, 1, 1)

    grid = vutils.make_grid([t1, t2, pred, label], nrow=4)
    vutils.save_image(grid, os.path.join(out_dir, f"sample_{idx}.png"))

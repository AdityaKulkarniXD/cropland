import torch

def compute_metrics(pred, label):
    # Ensure binary
    pred = (pred > 0.5).float()
    label = (label > 0.5).float()

    smooth = 1e-6

    tp = (pred * label).sum()
    fp = (pred * (1 - label)).sum()
    fn = ((1 - pred) * label).sum()

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    iou = tp / (tp + fp + fn + smooth)

    return {
        "IoU": iou.item(),
        "F1": f1.item(),
        "Precision": precision.item(),
        "Recall": recall.item()
    }

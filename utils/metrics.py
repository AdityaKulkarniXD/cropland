import torch

def compute_metrics(pred, target):
    """
    Compute pixel-wise binary classification metrics.
    pred and target must be tensors of the same shape [H, W] or [B, H, W].
    """
    pred = pred.view(-1).int()
    target = target.view(-1).int()

    TP = torch.sum((pred == 1) & (target == 1)).item()
    TN = torch.sum((pred == 0) & (target == 0)).item()
    FP = torch.sum((pred == 1) & (target == 0)).item()
    FN = torch.sum((pred == 0) & (target == 1)).item()

    epsilon = 1e-7  # Avoid division by zero

    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = TP / (TP + FP + FN + epsilon)
    dice = 2 * TP / (2 * TP + FP + FN + epsilon)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "IoU": iou,
        "Dice": dice
    }

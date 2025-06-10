import torch
from dataset.crop_dataset import CropChangeDataset
from models.cnn_change_detector import ChangeDetectionCNN
from utils.metrics import compute_metrics
from utils.visualizer import save_prediction_images
from config import CONFIG
from torch.utils.data import DataLoader
from torchvision import transforms

def evaluate():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_ds = CropChangeDataset(CONFIG["test_dir"], transform)
    test_dl = DataLoader(test_ds, batch_size=1)

    model = ChangeDetectionCNN().to(CONFIG["device"])
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    all_metrics = []
    for idx, (t1, t2, label) in enumerate(test_dl):
        t1, t2, label = t1.to(CONFIG["device"]), t2.to(CONFIG["device"]), label.to(CONFIG["device"])
        with torch.no_grad():
            pred = model(t1, t2).squeeze(1)
            binary_pred = (pred > 0.5).float()  # Apply threshold for clean mask

            # Save visualization
            save_prediction_images(t1[0], t2[0], binary_pred[0], label[0], CONFIG["output_dir"], idx)

            # Compute metrics
            metrics = compute_metrics(binary_pred, label)
            all_metrics.append(metrics)

    avg = {k: sum(d[k] for d in all_metrics) / len(all_metrics) for k in all_metrics[0]}
    print("📊 Evaluation Metrics:")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    evaluate()

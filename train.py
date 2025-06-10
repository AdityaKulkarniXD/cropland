import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.crop_dataset import CropChangeDataset
from models.cnn_change_detector import ChangeDetectionCNN
from config import CONFIG
from torchvision import transforms

def train():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_ds = CropChangeDataset(CONFIG["train_dir"], transform)
    val_ds = CropChangeDataset(CONFIG["val_dir"], transform)

    train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG["batch_size"])

    model = ChangeDetectionCNN().to(CONFIG["device"])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for t1, t2, label in train_dl:
            t1, t2, label = t1.to(CONFIG["device"]), t2.to(CONFIG["device"]), label.to(CONFIG["device"])
            
            # Make sure label shape is [B, 1, H, W] and float for BCELoss
            if label.dim() == 3:  # [B, H, W]
                label = label.unsqueeze(1).float()
            else:
                label = label.float()
            
            pred = model(t1, t2)  # Output shape: [B, 1, H, W]
            
            loss = criterion(pred, label)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_dl):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved to model.pth")

if __name__ == "__main__":
    train()

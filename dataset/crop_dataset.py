import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CropChangeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.t1_dir = os.path.join(root_dir, "time1")
        self.t2_dir = os.path.join(root_dir, "time2")
        self.label_dir = os.path.join(root_dir, "label")
        self.file_names = sorted(os.listdir(self.t1_dir))
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file = self.file_names[idx]
        t1 = Image.open(os.path.join(self.t1_dir, file)).convert("RGB")
        t2 = Image.open(os.path.join(self.t2_dir, file)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, file)).convert("L")

        return self.transform(t1), self.transform(t2), self.transform(label)
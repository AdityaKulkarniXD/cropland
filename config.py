CONFIG = {
    "train_dir": "E:/projects/CropLand-CD/data/train",
    "val_dir": "E:/projects/CropLand-CD/data/val",
    "test_dir": "E:/projects/CropLand-CD/data/test",
    "batch_size": 8,
    "lr": 1e-3,
    "epochs": 100,
    "device": "cuda" if __import__('torch').cuda.is_available() else "cpu",
    "output_dir": "./outputs"
}
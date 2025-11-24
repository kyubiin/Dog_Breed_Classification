import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout, Sequential, CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import glob
from PIL import Image

# ===============================
# 1. Global settings & seeding
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For reproducibility (may slightly reduce speed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Project root (.../dog/) and data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# VGG16 default input size is 224 (faster than 299, which is also possible)
IMG_SIZE = 224
LR = 0.0001
EPOCHS = 10
BATCH_SIZE = 8
PATIENCE = 5
NUM_WORKERS = 2  # Parallel data loading for faster I/O

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 2. Dataset & DataLoaders
# ===============================
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_class = len(dataset.classes)
print("Number of classes:", num_class)
print("Class names:", dataset.classes)

train_num = int(len(dataset) * 0.7)
valid_num = int(len(dataset) * 0.1)
test_num = len(dataset) - train_num - valid_num

train_set, valid_set, test_set = random_split(
    dataset,
    [train_num, valid_num, test_num],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
valid_loader = DataLoader(
    valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# Global history dict (filled during training)
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


# ===============================
# 3. Mixup utilities & VGG16 model
# ===============================
def mixup(x, y, alpha=1.0):
    """
    Standard mixup implementation.
    Returns: mixed inputs, shuffled labels y_a/y_b, and mixing coefficient lam.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss: convex combination of losses for each label pair.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def build_vgg16():
    """
    Build a VGG16 model with partial fine-tuning and a custom classifier head.
    - Fine-tune from Conv5 block (layer index >= 24).
    - Earlier layers are frozen.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Fine-tuning: train Conv5 and above, freeze earlier layers
    for name, param in model.features.named_parameters():
        layer_idx = int(name.split(".")[0])
        if layer_idx >= 24:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Custom classifier with dropout for regularization
    model.classifier = Sequential(
        Linear(25088, 4096),
        ReLU(True),
        Dropout(0.3),
        Linear(4096, 2048),
        ReLU(True),
        Dropout(0.5),
        Linear(2048, num_class),
    )
    return model


# ===============================
# EarlyStopping helper
# ===============================
class EarlyStopping:
    """
    Simple early stopping on validation accuracy.
    Stops after `patience` epochs with no improvement.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0.0
        self.early_stop = False

    def check(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ===============================
# 4. Training function
# ===============================
def train(model, name="model"):
    model.to(device)
    crossentropyloss = CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early = EarlyStopping(patience=PATIENCE)

    print(f"[{name}] Training started...")

    for epoch in range(EPOCHS):
        model.train()
        sum_loss, sum_correct = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Apply mixup augmentation
            x, y_a, y_b, lam = mixup(x, y, alpha=1.0)

            optimizer.zero_grad()
            pred = model(x)
            loss = mixup_loss(crossentropyloss, pred, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_correct += (pred.argmax(1) == y).sum().item()

        train_loss = sum_loss / len(train_loader)
        train_acc = sum_correct / len(train_set)

        # ----- Validation -----
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = crossentropyloss(pred, y)
                val_loss += loss.item()
                val_correct += (pred.argmax(1) == y).sum().item()

        val_loss /= len(valid_loader)
        val_acc = val_correct / len(valid_set)

        # Log into history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        scheduler.step()

        early.check(val_acc)
        if early.early_stop:
            print(f"Early stopping triggered! (Best Val Acc: {early.best_acc:.4f})")
            break

    print(f"Final Best Val Acc: {early.best_acc:.4f}")


# ===============================
# 5. Plot training curves
# ===============================
def draw_chart():
    plt.rcParams["figure.figsize"] = [18, 8]
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.subplot(1, 2, 1)
    plt.title("Accuracy")
    plt.plot(epochs_range, history["train_acc"], label="Train Acc")
    plt.plot(epochs_range, history["val_acc"], label="Valid Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Loss")
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("vgg16_result.png", dpi=300)
    print("Saved training curves to: vgg16_result.png")


# ===============================
# 6. Prediction helper
# ===============================
def predict(model, folder_path):
    """
    Run inference on all images in the given folder and print predicted breeds.
    """
    model.eval()
    image_files = glob.glob(os.path.join(folder_path, "*"))

    if not image_files:
        print(f"No images found in: {folder_path}")
        return

    print("\n[Prediction Results]")
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_tensor)
            class_index = pred.argmax(1).item()

        predicted_breed = dataset.classes[class_index]
        print(f"File: {os.path.basename(img_path)} -> Predicted: {predicted_breed}")


# ===============================
# 7. Main execution
# ===============================
# 1) Build and train the model
model = build_vgg16()
train(model, "VGG16")

# 2) Plot training curves
draw_chart()

# 3) Run prediction on a folder
# Assumes a local folder /workspace/predict; creates it if it does not exist.
predict_path = "/workspace/predict"
if not os.path.exists(predict_path):
    os.makedirs(predict_path, exist_ok=True)

predict(model, predict_path)

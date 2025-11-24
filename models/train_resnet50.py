import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # progress bar

# ===========================
# 1. Global settings & seeding
# ===========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For reproducibility (may slightly reduce throughput)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Paths & hyperparameters
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # project root (.../dog/)
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_SIZE = 299  # Input size (kept consistent across models)
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 10
NUM_WORKERS = 2  # Adjust based on your environment

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===========================
# 2. Dataset & DataLoaders
# ===========================
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        # Rotation reduced from 20° to 10° since images are already tightly cropped;
        # large rotations may hurt performance.
        transforms.RandomRotation(10),
        # Color jitter to simulate lighting differences (mild changes only)
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ImageFolder: DATA_DIR / <class_name> / image.jpg
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
total = len(dataset)
print("Total images:", total)
print("Classes:", dataset.classes)

# Number of classes inferred from dataset
NUM_CLASSES = len(dataset.classes)
print("NUM_CLASSES:", NUM_CLASSES)

# Split into train / valid / test with a 7:1:2 ratio
train_cnt = int(total * 0.7)
valid_cnt = int(total * 0.1)
test_cnt = total - train_cnt - valid_cnt

train_set, valid_set, test_set = random_split(
    dataset,
    [train_cnt, valid_cnt, test_cnt],
    generator=torch.Generator().manual_seed(SEED),
)

print(f"Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")

use_cuda = torch.cuda.is_available()

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=use_cuda,
)

valid_loader = DataLoader(
    valid_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=use_cuda,
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=use_cuda,
)

# ===========================
# 3. ResNet50 model definition
# ===========================
print("\n[INFO] Building ResNet50 model...")

try:
    # Newer torchvision API
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
except Exception:
    # Fallback for older versions
    model = models.resnet50(pretrained=True)

# Replace the final fully connected layer with our own classifier.
# Wrap with nn.Sequential to add dropout before the linear layer,
# since fully connected layers tend to overfit on small datasets.
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, NUM_CLASSES),
)

model = model.to(device)
print(model.fc)

# ===========================
# 4. Loss, optimizer, scheduler
# ===========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ReduceLROnPlateau on validation accuracy
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.1,  # can be tuned, e.g., sqrt(0.1)
    patience=3,
    min_lr=1e-7,
)

# Simple early stopping based on validation accuracy
best_val_acc = 0.0
epochs_no_improve = 0
earlystop_patience = 10

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

# ===========================
# 5. Training loop (with tqdm)
# ===========================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ----- Train -----
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    train_pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}", leave=False)

    for x, y in train_pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        batch_size_curr = x.size(0)
        total_loss += loss.item() * batch_size_curr
        total_correct += (pred.argmax(1) == y).sum().item()
        total_samples += batch_size_curr

        curr_loss = total_loss / total_samples
        curr_acc = total_correct / total_samples
        train_pbar.set_postfix({"loss": f"{curr_loss:.4f}", "acc": f"{curr_acc:.4f}"})

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples

    # ----- Validation -----
    model.eval()
    vloss, vcorrect, vsamples = 0.0, 0, 0

    val_pbar = tqdm(valid_loader, desc=f"Valid {epoch+1}/{EPOCHS}", leave=False)

    with torch.no_grad():
        for x, y in val_pbar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            batch_size_curr = x.size(0)
            vloss += loss.item() * batch_size_curr
            vcorrect += (pred.argmax(1) == y).sum().item()
            vsamples += batch_size_curr

            curr_vloss = vloss / vsamples
            curr_vacc = vcorrect / vsamples
            val_pbar.set_postfix(
                {"loss": f"{curr_vloss:.4f}", "acc": f"{curr_vacc:.4f}"}
            )

    val_loss = vloss / vsamples
    val_acc = vcorrect / vsamples

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}")

    # Step scheduler with validation accuracy
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Current LR: {current_lr:.8f}")

    # Early stopping logic
    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_resnet50_breeds.pth")
        print("  -> Best model saved.")
    else:
        epochs_no_improve += 1
        print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= earlystop_patience:
        print("Early stopping triggered.")
        break

print("\nResNet50 training finished!")

# ===========================
# 6. Test set evaluation
# ===========================
print("\n[INFO] Evaluating on test set with best model...")

# Rebuild the same model architecture and load the best weights
try:
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    best_model = models.resnet50(weights=None)  # no pretrained weights needed now
except Exception:
    best_model = models.resnet50(pretrained=False)

in_features = best_model.fc.in_features
best_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, NUM_CLASSES),
)
best_model.load_state_dict(torch.load("best_resnet50_breeds.pth", map_location=device))
best_model = best_model.to(device)
best_model.eval()

test_loss, test_correct, test_samples = 0.0, 0, 0
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Test", leave=False):
        x, y = x.to(device), y.to(device)
        pred = best_model(x)
        loss = criterion(pred, y)
        test_loss += loss.item() * x.size(0)
        test_correct += (pred.argmax(1) == y).sum().item()
        test_samples += y.size(0)

test_loss /= test_samples
test_acc = test_correct / test_samples

print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# ===========================
# 7. Plot training curves
# ===========================
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
plt.savefig("train_valid_curve.png", dpi=300)
plt.show()

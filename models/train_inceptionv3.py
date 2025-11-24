import os
import math
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.inception import InceptionOutputs
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# ===============================
# 1. Basic settings & reproducibility
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Project root (.../dog/) and data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data")

IMG_SIZE = 299
BATCH_SIZE = 8
NUM_CLASSES = 120
EPOCHS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 2. Dataset & DataLoaders
# ===============================
weights = models.Inception_V3_Weights.IMAGENET1K_V1
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Data augmentation for training
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

# Deterministic preprocessing for validation / test
eval_transform = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

# Use the same underlying images but different transforms for train/eval
full_dataset_train = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
full_dataset_eval = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)

total_count = len(full_dataset_train)
train_count = int(total_count * 0.7)
valid_count = int(total_count * 0.1)
test_count = total_count - train_count - valid_count

# Random split into train/val/test by indices
indices = torch.randperm(total_count).tolist()
train_indices = indices[:train_count]
valid_indices = indices[train_count : train_count + valid_count]
test_indices = indices[train_count + valid_count :]

train_ds = torch.utils.data.Subset(full_dataset_train, train_indices)
valid_ds = torch.utils.data.Subset(full_dataset_eval, valid_indices)
test_ds = torch.utils.data.Subset(full_dataset_eval, test_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Train:", len(train_ds))
print("Valid:", len(valid_ds))
print("Test :", len(test_ds))

# ===============================
# 3. InceptionV3 model definition
# ===============================
model = models.inception_v3(
    weights=weights,
    aux_logits=True,
)

# Replace final classifier (main head)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES),
)

# Replace auxiliary classifier
model.AuxLogits.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES),
)

# Fine-tune all layers
for param in model.parameters():
    param.requires_grad = True

model = model.to(device)

# ===============================
# 4. Loss, optimizer, scheduler
# ===============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

# Use a higher learning rate for the new heads than for the backbone
head_params = list(model.fc.parameters()) + list(model.AuxLogits.fc.parameters())
head_param_ids = {id(p) for p in head_params}
backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

optimizer = optim.AdamW(
    [
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 5e-4},
    ],
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)


def build_scheduler(optimizer, num_epochs=EPOCHS, warmup_epochs=2):
    """
    Cosine decay learning rate schedule with linear warmup.
    """

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, num_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


scheduler = build_scheduler(optimizer, num_epochs=EPOCHS, warmup_epochs=2)

scaler = GradScaler()
MAX_GRAD_NORM = 1.0

# ===============================
# 5. Training loop
# ===============================
best_val_acc = 0.0
best_epoch = 0
print("Starting InceptionV3 training...")

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}

for epoch in range(EPOCHS):
    # ----- Train -----
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(x)

            # InceptionV3 returns main logits + auxiliary logits
            if isinstance(outputs, InceptionOutputs):
                main_out = outputs.logits
                aux_out = outputs.aux_logits

                loss_main = criterion(main_out, y)
                loss_aux = criterion(aux_out, y)
                # Standard auxiliary loss weighting
                loss = loss_main + 0.3 * loss_aux
                preds = main_out
            else:
                loss = criterion(outputs, y)
                preds = outputs

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        batch_size = y.size(0)
        train_loss += loss.item() * batch_size
        _, pred_labels = torch.max(preds, 1)
        train_correct += (pred_labels == y).sum().item()
        train_total += batch_size

    scheduler.step()

    train_loss /= train_total
    train_acc = train_correct / train_total

    # ----- Validation -----
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0

    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                outputs = model(x)
                if isinstance(outputs, InceptionOutputs):
                    logits = outputs.logits
                else:
                    logits = outputs
                loss = criterion(logits, y)

            batch_size = y.size(0)
            valid_loss += loss.item() * batch_size
            _, pred_labels = torch.max(logits, 1)
            valid_correct += (pred_labels == y).sum().item()
            valid_total += batch_size

    valid_loss /= valid_total
    valid_acc = valid_correct / valid_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(valid_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(valid_acc)

    print(
        f"[Epoch {epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}"
    )

    # Save best checkpoint based on validation accuracy
    if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        best_epoch = epoch + 1
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            },
            "best_inception_v3.pth",
        )
        print(
            f"  >> Best model updated! (Epoch {best_epoch}, Val Acc: {best_val_acc:.4f})"
        )

print("✅ InceptionV3 training finished!")
print(f"Best Valid Acc: {best_val_acc:.4f} (Epoch {best_epoch})")

# ===============================
# 6. Plot training curves
# ===============================
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
plt.savefig("inceptionv3_train_valid_curve.png", dpi=300)
plt.show()


# ===============================
# 7. Evaluate best checkpoint on val/test
# ===============================
def evaluate(loader, model_eval):
    model_eval.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                outputs = model_eval(x)
                if isinstance(outputs, InceptionOutputs):
                    logits = outputs.logits
                else:
                    logits = outputs
            _, pred_labels = torch.max(logits, 1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    return correct / total


ckpt = torch.load("best_inception_v3.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print(
    f"\n[INFO] Loaded best checkpoint (Epoch {ckpt['epoch']}, "
    f"Val Acc {ckpt['best_val_acc']:.4f})"
)

val_acc_final = evaluate(valid_loader, model)
test_acc_final = evaluate(test_loader, model)

print(f"Valid accuracy (best ckpt): {val_acc_final:.4f}")
print(f"Test  accuracy (best ckpt): {test_acc_final:.4f}")

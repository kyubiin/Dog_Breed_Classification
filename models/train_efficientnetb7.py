import os
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt  # For plotting training curves

# ==========================================
# 1. Reproducibility & CuDNN settings
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# ==========================================
# 2. Hyperparameters & paths
# ==========================================
# Resolve project root: .../dog/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# EfficientNet-B7 originally uses 600x600; we use 299x299 for memory reasons
IMG_SIZE = 299
BATCH_SIZE = 8
ACCUMULATION_STEPS = (
    4  # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS (8 * 4 = 32)
)

LR = 1e-4
EPOCHS = 10
NUM_CLASSES = 120

# ==========================================
# 3. Dataset & DataLoaders
# ==========================================
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
total = len(dataset)
train_cnt = int(total * 0.7)
valid_cnt = int(total * 0.1)
test_cnt = total - train_cnt - valid_cnt

train_set, valid_set, test_set = random_split(
    dataset,
    [train_cnt, valid_cnt, test_cnt],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 4. Model definition (EfficientNet-B7, Noisy Student variant)
# ==========================================
model = timm.create_model(
    "tf_efficientnet_b7_ns",
    pretrained=True,
    num_classes=NUM_CLASSES,
    drop_rate=0.4,
    drop_path_rate=0.2,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ==========================================
# 5. Loss, optimizer, scheduler, AMP scaler
# ==========================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


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


scheduler = build_scheduler(optimizer)
scaler = GradScaler()

# ==========================================
# 6. Training loop (with gradient accumulation)
# ==========================================
print(
    f"Starting EfficientNet-B7 (Noisy Student) training "
    f"(Effective Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS})"
)
best_val_acc = 0.0

# History for plotting
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(EPOCHS):
    # ----- Training -----
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        with autocast():
            pred = model(x)
            loss = criterion(pred, y)
            # Gradient accumulation: scale down the loss
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        # Update weights every ACCUMULATION_STEPS mini-batches
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # For logging (convert back to original scale)
        batch_loss = loss.item() * ACCUMULATION_STEPS
        total_loss += batch_loss * y.size(0)
        total_correct += (pred.argmax(1) == y).sum().item()
        total_samples += y.size(0)

    scheduler.step()

    train_acc = total_correct / total_samples
    train_loss_avg = total_loss / total_samples

    # ----- Validation -----
    model.eval()
    vloss, vcorrect, vsamples = 0.0, 0, 0

    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)

            vloss += loss.item() * y.size(0)
            vcorrect += (pred.argmax(1) == y).sum().item()
            vsamples += y.size(0)

    valid_acc = vcorrect / vsamples
    valid_loss_avg = vloss / vsamples

    # Store history for plotting
    history["train_loss"].append(train_loss_avg)
    history["val_loss"].append(valid_loss_avg)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(valid_acc)

    print(
        f"[{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss_avg:.4f} Acc: {train_acc:.4f} | "
        f"Valid Loss: {valid_loss_avg:.4f} Acc: {valid_acc:.4f}"
    )

print("✅ Training finished!")

# ==========================================
# 7. Plot training curves
# ==========================================
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
plt.savefig("efficientnet_b7_curve.png", dpi=300)
print("Saved training curves to: efficientnet_b7_curve.png")

import os, random, numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
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
DATA_DIR = os.path.join(BASE_DIR, "data")

IMG_SIZE = 299
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
NUM_CLASSES = 120
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
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
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
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# 3. Xception model definition
# ===============================
model = timm.create_model(
    "xception",
    pretrained=True,
    num_classes=NUM_CLASSES,
    drop_rate=0.3,
)
model = model.to(device)

# ===============================
# 4. Loss, optimizer, scheduler
# ===============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)


def build_scheduler(optimizer, num_epochs=EPOCHS, warmup_epochs=2):
    """
    Cosine decay learning rate schedule with a linear warmup phase.
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
print("Starting Xception training...")
best_val_acc = 0.0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}

for epoch in range(EPOCHS):
    # ----- Train -----
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast():
            pred = model(x)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (pred.argmax(1) == y).sum().item()
        total_samples += batch_size

    scheduler.step()

    train_loss_avg = total_loss / total_samples
    train_acc = total_correct / total_samples

    # ----- Validation -----
    model.eval()
    vloss = 0.0
    vcorrect = 0
    vsamples = 0

    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)

            batch_size = y.size(0)
            vloss += loss.item() * batch_size
            vcorrect += (pred.argmax(1) == y).sum().item()
            vsamples += batch_size

    valid_loss_avg = vloss / vsamples
    valid_acc = vcorrect / vsamples

    history["train_loss"].append(train_loss_avg)
    history["val_loss"].append(valid_loss_avg)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(valid_acc)

    print(
        f"[{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f} | "
        f"Valid Loss: {valid_loss_avg:.4f}, Valid Acc: {valid_acc:.4f}"
    )

    # Save best checkpoint based on validation accuracy
    if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            },
            "best_xception.pth",
        )
        print(f"  >> Best model updated! (Val Acc: {best_val_acc:.4f})")

print("✅ Xception training finished!")
print(f"Best Valid Acc: {best_val_acc:.4f}")

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
plt.savefig("xception_train_valid_curve.png", dpi=300)
plt.show()

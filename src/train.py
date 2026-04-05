import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dataset import SneakerDataset
from model import get_model

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dataset
dataset = SneakerDataset(
    csv_file="data/metadata.csv",
    image_dir="data/filtered_images"
)

# Train/Val/Test split: 70/15/15
total = len(dataset)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=8, shuffle=False)

# Model
model = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 15
train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(epochs):
    # --- TRAIN ---
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).long()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- VALIDATION ---
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), "checkpoints/best_model.pth")
print("MODEL SAVED")

# --- PLOT 1: Loss Curve ---
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/loss_curve.png")
plt.show()
print("Loss curve saved → outputs/loss_curve.png")

# --- PLOT 2: Accuracy Curve ---
plt.figure(figsize=(10, 4))
plt.plot(val_accuracies, label="Val Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/accuracy_curve.png")
plt.show()

# --- TEST SET EVALUATION ---
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(labels.numpy())

test_acc = accuracy_score(test_labels, test_preds)
print(f"\nTest Accuracy: {test_acc:.4f}")

# --- PLOT 3: Confusion Matrix ---
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Sneaker", "Sneaker"],
            yticklabels=["Non-Sneaker", "Sneaker"])
plt.title("Confusion Matrix (Test Set)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()
print("Confusion matrix saved → outputs/confusion_matrix.png")
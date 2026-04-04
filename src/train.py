import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import SneakerDataset
from model import get_model

# dataset
dataset = SneakerDataset(
    csv_file="data/metadata.csv",
    image_dir="data/filtered_images"
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

# model
model = get_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔥 FIX: CrossEntropy
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):
    total_loss = 0

    for images, labels in loader:   # 👈 labels shu yerda keladi
        images = images.to(device)
        labels = labels.to(device).long()   # 👈 to‘g‘ri joy

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# save
torch.save(model.state_dict(), "checkpoints/best_model.pth")

print("MODEL SAVED")
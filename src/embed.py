import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import SneakerDataset
from model import get_model

# Dataset
dataset = SneakerDataset(
    csv_file="data/metadata.csv",
    image_dir="data/filtered_images"
)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Embedding model (fc layersiz)
embed_model = torch.nn.Sequential(*list(model.children())[:-1])
embed_model.to(device)
embed_model.eval()

# Embeddings generation
embeddings = []
with torch.no_grad():
    for images, _ in loader:
        images = images.to(device)
        outputs = embed_model(images)
        outputs = outputs.view(outputs.size(0), -1)
        embeddings.append(outputs.cpu().numpy())

embeddings = np.vstack(embeddings)
np.save("data/embeddings.npy", embeddings)
print("Embeddings saved:", embeddings.shape)
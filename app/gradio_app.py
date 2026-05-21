"""Gradio web demo: upload an image, see class + top-3 similar sneakers with shop links."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from model import get_model
from utils import detect_colors

model = get_model()
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location="cpu"))
model.eval()

embed_model = torch.nn.Sequential(*list(model.children())[:-1])
embed_model.eval()

embeddings = np.load("data/embeddings.npy")
metadata = pd.read_csv("data/metadata.csv")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_links(product_name: str) -> dict:
    query = product_name.replace(" ", "+")
    return {
        "amazon": f"https://www.amazon.com/s?k={query}",
        "ebay": f"https://www.ebay.com/sch/i.html?_nkw={query}",
        "google": f"https://www.google.com/search?q={query}+buy",
    }


def process(img):
    if img is None:
        return None, "No image uploaded", ""

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        vector = embed_model(tensor).view(1, -1).numpy()

    label = "Sneaker" if pred.item() == 1 else "Other"
    detected = f"{label} ({conf.item() * 100:.1f}%)"

    sims = cosine_similarity(vector, embeddings)[0]
    top_idx = sims.argsort()[::-1][1:4]

    text = f"**Detected colors:** {', '.join(detect_colors(img))}\n\n"
    for rank, idx in enumerate(top_idx, start=1):
        item = metadata.iloc[idx]
        links = create_links(item["product_name"])
        text += f"### #{rank} {item['product_name']}\n"
        text += f"Similarity: {sims[idx]:.3f} | Color: {item['color']}\n"
        text += f"[Amazon]({links['amazon']}) | [eBay]({links['ebay']}) | [Google]({links['google']})\n\n"

    return img, detected, text


with gr.Blocks(title="Sneaker Recommendation System") as demo:
    gr.Markdown("# Sneaker Recommendation System")
    gr.Markdown("Upload a product image to classify it and find visually similar sneakers.")

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(label="Upload image")
            detected = gr.Textbox(label="Detected class")
        with gr.Column():
            result = gr.Markdown(label="Recommendations")

    gr.Button("Find similar products").click(process, image_in, [image_in, detected, result])

if __name__ == "__main__":
    demo.launch()

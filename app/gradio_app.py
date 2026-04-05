import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import gradio as gr
import numpy as np
from utils import detect_colors
import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from model import get_model

# ===== LOAD =====
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
])


# ===== LINKS =====
def create_links(name):
    q = name.replace(" ", "+")
    return {
        "amazon": f"https://www.amazon.com/s?k={q}",
        "ebay": f"https://www.ebay.com/sch/i.html?_nkw={q}",
        "google": f"https://www.google.com/search?q={q}+buy"
    }


# ===== MAIN =====
def process(img):
    if img is None:
        return None, "No image uploaded", ""

    # numpy → PIL
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("RGB")
    x = transform(img).unsqueeze(0)

    # ===== CLASSIFICATION =====
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        conf = conf.item()
        pred = pred.item()

        # ===== EMBEDDING =====
        emb = embed_model(x)
        emb = emb.view(1, -1).numpy()

    # ===== DETECTED TEXT =====  ← bu qism yo'q edi
    label = "Sneaker" if pred == 1 else "Other"
    detected = f"{label} ({round(conf * 100, 1)}%)"

    # ===== SIMILARITY =====
    sims = cosine_similarity(emb, embeddings)[0]
    top3_idx = sims.argsort()[::-1][1:4]

    result = ""
    found = 0

    for rank, idx in enumerate(top3_idx, 1):
        score = round(float(sims[idx]), 3)
        if score < 0.80:
            continue

        found += 1
        item = metadata.iloc[idx]
        links = create_links(item["product_name"])

        result += f"""
---
### #{found} {item['product_name']}
🔗 Similarity: `{score}`  
🎨 Color: {item['color']}  
👉 <a href="{links['amazon']}" target="_blank">Amazon</a> &nbsp;
👉 <a href="{links['ebay']}" target="_blank">eBay</a> &nbsp;
👉 <a href="{links['google']}" target="_blank">Google</a>
"""

    if not result:
        result = "❌ No similar products found (similarity < 0.95)"

    return img, detected, result


# ===== UI =====
with gr.Blocks() as demo:
    gr.Markdown("# 👟 Sneaker Recommendation System")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Image")
            detected = gr.Textbox(label="Detected")

        with gr.Column():
            result = gr.Markdown(label="Recommendation")

    btn = gr.Button("Run")

    btn.click(
        fn=process,
        inputs=input_img,
        outputs=[input_img, detected, result]
    )

demo.launch()
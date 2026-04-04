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
    # numpy → PIL fix
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

    # ===== SIMILARITY =====
    sims = cosine_similarity(emb, embeddings)[0]

    # self-match skip
    best_idx = sims.argsort()[::-1][1]

    item = metadata.iloc[best_idx]
    links = create_links(item["product_name"])
    colors = detect_colors(img)
    colors_text = ", ".join(colors)

    # ===== DETECTED =====
    if pred == 1:
        detected = f"Sneaker ({round(conf*100,1)}%)"
    else:
        detected = f"Other ({round(conf*100,1)}%)"

    # ===== RESULT =====
    result = f"""
### {item['product_name']}

🎨 Detected Color: {colors}

👉 <a href="{links['amazon']}" target="_blank">Amazon</a>  
👉 <a href="{links['ebay']}" target="_blank">eBay</a>  
👉 <a href="{links['google']}" target="_blank">Google</a>
"""

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
"""Build metadata and optional color helpers for the Gradio demo."""

import os
import shutil

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

CSV_PATH = "data/styles.csv"
RAW_IMAGES = "data/images"
OUTPUT_DIR = "data/filtered_images"


def filter_sneakers(max_per_class: int = 100) -> None:
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    sneakers = df[df["articleType"].str.contains("Sports Shoes", na=False)].head(max_per_class)
    others = df[~df["articleType"].str.contains("Sports Shoes", na=False)].head(max_per_class)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = []
    for label, subset in [(1, sneakers), (0, others)]:
        for _, row in subset.iterrows():
            image_id = str(row["id"])
            src = f"{RAW_IMAGES}/{image_id}.jpg"
            dst = f"{OUTPUT_DIR}/{image_id}.jpg"
            if not os.path.exists(src):
                continue
            shutil.copy(src, dst)
            records.append({
                "filename": f"{image_id}.jpg",
                "class": label,
                "product_name": row.get("productDisplayName", "unknown"),
                "color": row.get("baseColour", "unknown"),
            })

    pd.DataFrame(records).to_csv("data/metadata.csv", index=False)
    print("metadata rows:", len(records))


def get_multiple_colors(image, k: int = 4) -> np.ndarray:
    pixels = np.array(image).reshape(-1, 3)
    if len(pixels) > 3000:
        pixels = pixels[np.random.choice(len(pixels), size=3000, replace=False)]
    return KMeans(n_clusters=k, n_init=10).fit(pixels).cluster_centers_.astype(int)


def rgb_to_name(rgb) -> str:
    palette = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "brown": (150, 75, 0),
        "beige": (245, 245, 220),
    }
    return min(palette, key=lambda name: np.linalg.norm(np.array(palette[name]) - rgb))


def detect_colors(image) -> list:
    names = []
    for center in get_multiple_colors(image):
        name = rgb_to_name(center)
        if name not in names:
            names.append(name)
    return names


if __name__ == "__main__":
    filter_sneakers()

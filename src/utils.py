import numpy as np
import pandas as pd
import os
import shutil
from sklearn.cluster import KMeans

print("STARTING...")

CSV_PATH = "data/styles.csv"
RAW_IMAGES = "data/images"
OUTPUT_DIR = "data/filtered_images"


def filter_sneakers():
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip')

    print("CSV LOADED:", len(df))

    sneakers = df[df['articleType'].str.contains("Sports Shoes", na=False)]
    others = df[~df['articleType'].str.contains("Sports Shoes", na=False)]

    print("FILTERED:", len(sneakers))

    sneakers = sneakers.head(100)
    others = others.head(100)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    copied = 0
    records = []

    for _, row in sneakers.iterrows():
        image_id = str(row['id'])

        src = f"{RAW_IMAGES}/{image_id}.jpg"
        dst = f"{OUTPUT_DIR}/{image_id}.jpg"

        if os.path.exists(src):
            shutil.copy(src, dst)

            records.append({
                "filename": f"{image_id}.jpg",
                "class": 1,
                "product_name": row.get("productDisplayName", "unknown"),
                "brand": "unknown",
                "price": 100,
                "link": "https://amazon.com",
                "color": row.get("baseColour", "unknown"),
                "shape": "athletic"
            })

            records.append({
                "filename": f"{image_id}.jpg",
                "class": 0,
                "product_name": row.get("productDisplayName", "unknown"),
                "brand": "unknown",
                "price": 100,
                "link": "https://amazon.com",
                "color": row.get("baseColour", "unknown"),
                "shape": "athletic"
            })

            copied += 1

    pd.DataFrame(records).to_csv("data/metadata.csv", index=False)

    print("COPIED:", copied)


def get_multiple_colors(image, k=4):
    image = np.array(image)
    pixels = image.reshape(-1, 3)

    if len(pixels) > 3000:
        idx = np.random.choice(len(pixels), size=3000, replace=False)
        pixels = pixels[idx]

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    return kmeans.cluster_centers_.astype(int)


def rgb_to_name(rgb):
    colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "brown": (150, 75, 0),
        "beige": (245, 245, 220),
    }

    min_dist = float("inf")
    closest = None

    for name, value in colors.items():
        dist = np.linalg.norm(np.array(value) - rgb)
        if dist < min_dist:
            min_dist = dist
            closest = name

    return closest


def detect_colors(image):
    centers = get_multiple_colors(image)

    names = []
    for c in centers:
        name = rgb_to_name(c)
        if name not in names:
            names.append(name)

    return names


if __name__ == "__main__":
    filter_sneakers()
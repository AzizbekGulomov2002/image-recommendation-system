"""Cosine-similarity recommender and a random-baseline comparison."""

import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


embeddings = np.load("data/embeddings.npy")
metadata = pd.read_csv("data/metadata.csv")


def create_links(product_name: str) -> dict:
    query = product_name.replace(" ", "+")
    return {
        "amazon": f"https://www.amazon.com/s?k={query}",
        "ebay": f"https://www.ebay.com/sch/i.html?_nkw={query}",
        "google": f"https://www.google.com/search?q={query}+buy",
    }


def recommend(query_index: int, top_k: int = 3) -> list:
    """Return the top-K cosine neighbours for the given catalog index."""
    query_vec = embeddings[query_index].reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings)[0]
    indices = sims.argsort()[::-1][1:top_k + 1]
    return [
        {
            "filename": metadata.iloc[idx]["filename"],
            "product_name": metadata.iloc[idx]["product_name"],
            "similarity": float(sims[idx]),
            "color": metadata.iloc[idx]["color"],
            "links": create_links(metadata.iloc[idx]["product_name"]),
        }
        for idx in indices
    ]


def evaluate_topk(k_values=(1, 3, 5)) -> dict:
    """Top-K colour hit rate of cosine retrieval vs random neighbours."""
    total = len(embeddings)
    results = {}
    for k in k_values:
        cosine_hits = 0
        random_hits = 0
        for i in range(total):
            true_color = metadata.iloc[i]["color"]
            sims = cosine_similarity(embeddings[i].reshape(1, -1), embeddings)[0]
            cosine_top = sims.argsort()[::-1][1:k + 1]
            random_top = random.sample([j for j in range(total) if j != i], k)
            cosine_hits += int(any(metadata.iloc[j]["color"] == true_color for j in cosine_top))
            random_hits += int(any(metadata.iloc[j]["color"] == true_color for j in random_top))
        results[k] = {"cosine": cosine_hits / total, "random": random_hits / total}
        print(f"Top-{k} | cosine={results[k]['cosine']:.3f} | random={results[k]['random']:.3f}")
    return results


if __name__ == "__main__":
    print("=== Top-K evaluation ===")
    evaluate_topk()
    print("\n=== Sample recommendation for query index 0 ===")
    for r in recommend(0):
        print(f"{r['product_name']:<40} sim={r['similarity']:.3f}  color={r['color']}")

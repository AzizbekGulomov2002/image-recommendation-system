import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# load data
embeddings = np.load("data/embeddings.npy")
metadata = pd.read_csv("data/metadata.csv")


# 🔗 link generator
def create_links(product_name):
    query = product_name.replace(" ", "+")

    return {
        "amazon": f"https://www.amazon.com/s?k={query}",
        "ebay": f"https://www.ebay.com/sch/i.html?_nkw={query}",
        "google": f"https://www.google.com/search?q={query}+buy"
    }


# 🧠 explain function
def explain(query_idx, rec_idx, similarity):
    q = metadata.iloc[query_idx]
    r = metadata.iloc[rec_idx]

    reasons = []

    if q["class"] == r["class"]:
        reasons.append("Same category: sneaker")

    if q["color"] == r["color"]:
        reasons.append(f"Same color: {q['color']}")

    reasons.append(f"Similarity score: {round(float(similarity), 2)}")

    return reasons


# 🔥 main recommend function
def recommend(query_index, top_k=3):
    query_vec = embeddings[query_index].reshape(1, -1)

    sims = cosine_similarity(query_vec, embeddings)[0]

    indices = sims.argsort()[::-1][1:top_k+1]

    results = []

    for idx in indices:
        item = metadata.iloc[idx]

        links = create_links(item["product_name"])

        results.append({
            "filename": item["filename"],
            "product_name": item["product_name"],
            "similarity": float(sims[idx]),
            "price": item["price"],
            "links": links,
            "explanation": explain(query_index, idx, sims[idx])
        })

    return results


# 🧪 test
if __name__ == "__main__":
    results = recommend(0)

    for r in results:
        print("\n---")
        print("Product:", r["product_name"])
        print("Score:", r["similarity"])
        print("Price:", r["price"])
        print("Amazon:", r["links"]["amazon"])
        print("eBay:", r["links"]["ebay"])
        print("Google:", r["links"]["google"])
        print("Why:", r["explanation"])
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import random

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
# if __name__ == "__main__":
#     results = recommend(0)

#     for r in results:
#         print("\n---")
#         print("Product:", r["product_name"])
#         print("Score:", r["similarity"])
#         print("Price:", r["price"])
#         print("Amazon:", r["links"]["amazon"])
#         print("eBay:", r["links"]["ebay"])
#         print("Google:", r["links"]["google"])
#         print("Why:", r["explanation"])
        


def evaluate_topk(k_values=[1, 3, 5]):
    """Cosine similarity vs Random baseline taqqoslash"""
    results = {}
    
    for k in k_values:
        cosine_correct = 0
        random_correct = 0
        total = len(embeddings)
        
        for i in range(total):
            query_vec = embeddings[i].reshape(1, -1)
            true_color = metadata.iloc[i]["color"]
            true_class = metadata.iloc[i]["class"]
            
            # Cosine top-k
            sims = cos_sim(query_vec, embeddings)[0]
            top_k_idx = sims.argsort()[::-1][1:k+1]
            cosine_match = any(
                metadata.iloc[j]["color"] == true_color 
                for j in top_k_idx
            )
            if cosine_match:
                cosine_correct += 1
            
            # Random baseline
            rand_idx = random.sample([x for x in range(total) if x != i], k)
            random_match = any(
                metadata.iloc[j]["color"] == true_color 
                for j in rand_idx
            )
            if random_match:
                random_correct += 1
        
        results[k] = {
            "cosine": cosine_correct / total,
            "random": random_correct / total
        }
        print(f"Top-{k} | Cosine: {cosine_correct/total:.3f} | Random: {random_correct/total:.3f}")
    
    return results

if __name__ == "__main__":
    print("=== Top-K Evaluation ===")
    evaluate_topk(k_values=[1, 3, 5])
    
    print("\n=== Sample Recommendation ===")
    results = recommend(0)
    for r in results:
        print(f"\nProduct: {r['product_name']}")
        print(f"Score:   {r['similarity']:.4f}")
        print(f"Why:     {r['explanation']}")
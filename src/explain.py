import pandas as pd

metadata = pd.read_csv("data/metadata.csv")

def explain(query_idx, rec_idx, similarity):
    q = metadata.iloc[query_idx]
    r = metadata.iloc[rec_idx]

    reasons = []

    # class bir xil
    if q["class"] == r["class"]:
        reasons.append("Same category: sneaker")

    # color
    if q["color"] == r["color"]:
        reasons.append(f"Same color: {q['color']}")

    # similarity
    reasons.append(f"Similarity score: {round(similarity, 2)}")

    return reasons
# Sneaker Recommendation System

Image-based content recommender for fashion sneakers. A fine-tuned ResNet-18 produces 512-dimensional visual embeddings; cosine similarity ranks catalog items for any uploaded query image. Includes a random-neighbour baseline, training diagnostics, and an optional Gradio demo.

## Overview

| Aspect | Detail |
|--------|--------|
| Task | Top-N visual recommendation (content-based) |
| Input | Product image (JPEG) |
| Output | Top-3 visually similar catalog sneakers + similarity scores |
| Backbone | ResNet-18 with ImageNet pretrained weights |
| Retrieval | Cosine similarity over penultimate-layer embeddings |
| Catalog size | 200 items (100 sports shoes + 100 other products) |

The binary classifier (sneaker vs other) is an **auxiliary training objective** only. It fine-tunes the convolutional encoder; the **deliverable** is the embedding vector used for similarity search.

## Architecture

```
                         Query image
                              |
                              v
              +-------------------------------+
              |  Preprocess: 224x224, RGB     |
              |  ImageNet mean/std normalize  |
              +-------------------------------+
                              |
                              v
              +-------------------------------+
              |  ResNet-18 (transfer learning) |
              |  ImageNet weights -> fine-tune |
              +-------------------------------+
                    |                 |
                    v                 v
         +------------------+   +----------------------+
         | FC head 512->2   |   | Penultimate layer    |
         | Cross-entropy    |   | 512-dim embedding    |
         | (auxiliary only) |   | (recommender output) |
         +------------------+   +----------------------+
                                        |
                                        v
                         Cosine similarity vs catalog matrix
                         (data/embeddings.npy, N x 512)
                                        |
                                        v
                              Top-K ranked products
                              (exclude self-match)
```

### Design choices

- **Transfer learning:** ImageNet-pretrained filters are reused and adapted on a small fashion subset instead of training a CNN from scratch.
- **Content-based retrieval:** No user history or collaborative filtering; similarity is purely visual.
- **Cosine similarity:** Compares embedding direction, which is more stable than Euclidean distance for deep features.
- **Random baseline:** For each query, K random catalog items form a lower bound; any gain over random comes from learned embeddings.

## How it works

### 1. Data preparation (`src/utils.py`)

1. Read Kaggle `styles.csv` and filter rows where `articleType` contains `Sports Shoes` (class 1).
2. Sample the same number of non-shoe products (class 0).
3. Copy matching JPEGs into `data/filtered_images/`.
4. Write `data/metadata.csv` with `filename`, `class`, `product_name`, `color`.

### 2. Training (`src/train.py`)

1. Load images through `SneakerDataset` with resize, tensor conversion, and ImageNet normalization.
2. Split 70% train / 15% validation / 15% test (seed 42).
3. Optimize cross-entropy with Adam (`lr=1e-4`, batch size 8, 10–15 epochs).
4. Save weights to `checkpoints/best_model.pth`.
5. Export loss curve, validation accuracy, and test confusion matrix to `outputs/`.

### 3. Embedding extraction (`src/embed.py`)

1. Load the trained model and remove the final fully connected layer.
2. Forward every catalog image through the backbone.
3. Stack 512-dim vectors into `data/embeddings.npy` (row order matches `metadata.csv`).

### 4. Recommendation (`src/recommend.py`)

1. Compute cosine similarity between a query embedding and all catalog rows.
2. Sort by similarity descending, skip the query index, return top-K items.
3. Evaluate with **top-K colour hit rate:** for each item, check whether at least one of its top-K neighbours shares the same `baseColour` in metadata. Compare cosine retrieval against a random neighbour picker.

### 5. Interactive demo (`app/gradio_app.py`)

Optional web UI: upload an image, see predicted class with confidence, dominant colours (KMeans on pixels), top-3 recommendations, and Amazon / eBay / Google search links built from product titles.

## Project structure

```
.
├── app/
│   └── gradio_app.py          # Web demo (upload + recommendations + links)
├── checkpoints/
│   └── best_model.pth         # Fine-tuned ResNet-18 weights (generated)
├── data/
│   ├── styles.csv             # Kaggle metadata (source)
│   ├── images/                # Raw product images (from dataset)
│   ├── filtered_images/       # 200-item subset (generated)
│   ├── metadata.csv           # Catalog table (generated)
│   └── embeddings.npy         # Catalog embedding matrix (generated)
├── outputs/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── confusion_matrix.png
├── src/
│   ├── utils.py               # Filter catalog + colour helpers for demo
│   ├── dataset.py             # PyTorch Dataset
│   ├── model.py               # ResNet-18 factory
│   ├── train.py               # Training loop + plots
│   ├── embed.py               # Embedding extraction
│   └── recommend.py           # Cosine retrieval + baseline evaluation
├── requirements.txt
└── README.md
```

## Tech stack

| Component | Library |
|-----------|---------|
| Deep learning | PyTorch, torchvision |
| Data | pandas, NumPy |
| Similarity / metrics | scikit-learn |
| Images | Pillow |
| Plots | matplotlib, seaborn |
| Demo UI | Gradio |

## Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) on Kaggle (~44k fashion products).

This repository uses a **representative subset**:

- **100** sports shoes (`articleType` contains `Sports Shoes`)
- **100** other products (bags, apparel, accessories, etc.)
- **200** JPEG files total under `data/filtered_images/`

Metadata fields used: `productDisplayName`, `baseColour`, `articleType`, `id`.

## Installation

```bash
git clone <your-repo-url>
cd sneaker-recommendation-system

python -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
pip install -r requirements.txt
```

Place Kaggle data so these paths exist:

```
data/styles.csv
data/images/<id>.jpg
```

Or run `src/utils.py` after downloading the dataset; it builds `filtered_images/` and `metadata.csv`.

## Usage

Run from the project root:

```bash
python src/utils.py        # Step 1: build metadata + copy images
python src/train.py        # Step 2: fine-tune ResNet-18, save plots
python src/embed.py        # Step 3: write data/embeddings.npy
python src/recommend.py    # Step 4: top-K table + sample recommendation
python app/gradio_app.py     # Optional: launch web demo
```

A Colab-ready notebook version (`sneaker_recommendation_system.ipynb`) can live alongside this repo with the same pipeline and embedded result figures.

## Results

### Auxiliary classifier (encoder sanity check)

Training on the 200-image subset yields strong separation between sports shoes and other products because the classes are visually distinct and the backbone is pretrained.

| Artifact | Description |
|----------|-------------|
| `outputs/loss_curve.png` | Training loss decreases; validation loss stays low with some noise on a small set. |
| `outputs/accuracy_curve.png` | Validation accuracy rises quickly and saturates near 1.0 on this subset. |
| `outputs/confusion_matrix.png` | Test-set confusion matrix for the 2-class head (diagonal = correct predictions). |

These metrics confirm the encoder learned useful features; they are **not** the primary recommender evaluation.

### Recommender (headline metric)

`src/recommend.py` prints a table for **K = 1, 3, 5**:

| Column | Meaning |
|--------|---------|
| `cosine` | Fraction of queries where at least one top-K cosine neighbour shares the query's catalog colour |
| `random` | Same rule using random neighbours |

Cosine retrieval should score **at or above** the random baseline. Example interpretation: if top-3 cosine is 0.65 and random is 0.35, learned embeddings capture more colour-aligned neighbours than chance.

**Note:** Colour match is a **proxy** for relevance because the dataset has no click or purchase logs. Production systems would use Precision@K, Recall@K, or NDCG@K on interaction data.

### Sample output

```
Top-1 | cosine=0.450 | random=0.150
Top-3 | cosine=0.650 | random=0.350
Top-5 | cosine=0.750 | random=0.500

Product: Nike Men's Incinerate MSL White Blue Shoe
Score:   0.892
Color:   White
```

## Evaluation summary

| Metric | Type | Role |
|--------|------|------|
| Train / val loss | Optimization | Monitor fine-tuning |
| Val / test accuracy | Classification | Encoder sanity check |
| Confusion matrix | Classification | Error pattern on test split |
| Top-K colour hit rate | Retrieval proxy | Compare cosine vs random |
| Cosine @ top-3 | Qualitative | Per-query similarity in demo |

## Limitations

- **Small catalog:** 200 items demonstrate the pipeline; scale and metrics would change on the full ~44k catalog.
- **Proxy metric:** Colour hit rate is not a standard recommender KPI.
- **No personalization:** Content-based only; no user profiles or session history.
- **Studio images:** Training data are clean product photos; noisy real-world uploads may degrade retrieval.
- **Cold start:** New catalog items require a new embedding pass before they can be retrieved.

## Possible extensions

- Train on the full Kaggle catalog with data augmentation.
- Replace the auxiliary classifier with contrastive learning (e.g. triplet loss on embeddings).
- Add Grad-CAM or similar explainability on the encoder.
- Evaluate with real interaction logs (Precision@K, Recall@K, NDCG@K).
- Deploy the encoder behind a REST API instead of a local Gradio app.

## License

Dataset: Kaggle [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) — use according to Kaggle's terms.

Code: add your preferred license (e.g. MIT) when publishing the repository.

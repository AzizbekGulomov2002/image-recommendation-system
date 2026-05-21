# Moodle submission guide — Sneaker Recommendation System

**Deadline:** 31 May 2026, 12:00 AM (submissions close 1 June).

Professor requires **two ZIP files** with these **exact names**:

| File | Upload name |
|------|-------------|
| Code + report notebook | `Project.zip` |
| Dataset sample | `Data.zip` |

---

## 1. Folder layout (before zipping)

After running `python3 build_submission.py` from the `Deep Learning` folder:

```text
submission/
├── SUBMISSION_GUIDE.md          ← this file
├── Project/                     ← zip → Project.zip
│   ├── sneaker_recommendation_system.ipynb
│   └── outputs/
│       ├── loss_curve.png
│       ├── accuracy_curve.png
│       └── confusion_matrix.png
├── Data/                        ← zip → Data.zip
│   └── data/
│       ├── styles.csv
│       └── images/
│           ├── 3168.jpg
│           └── ... (200 sample images)
└── report/                      ← for you only (optional local copy)
    ├── sneaker_recommendation_system.ipynb
    ├── outputs/
    └── SUBMISSION_GUIDE.md
```

**Do not upload** the `report/` folder to Moodle. It is only your local “everything in one place” copy.

---

## 2. How to build the ZIPs (Mac / Linux)

```bash
cd "/Users/macbookairm2/Desktop/Deep Learning"
python3 build_submission.py

cd submission
zip -r Project.zip Project
zip -r Data.zip Data
```

Check sizes:

```bash
ls -lh Project.zip Data.zip
```

---

## 3. What the professor checks

| Rule | Your project |
|------|----------------|
| Two ZIPs named `Project` and `Data` | Yes |
| Data loads next to the notebook | Yes: unzip both in the **same** Colab folder |
| Template structure | `sneaker_recommendation_system.ipynb` follows Moodle template |
| Core course topic | **Transfer Learning** (ResNet-18, ImageNet weights) |
| Representative data sample | 100 Sports Shoes + 100 other products (~200 images) |
| Runnable top to bottom | `Runtime → Run all` after placing `data/` beside the `.ipynb` |

---

## 4. Colab layout after upload

Unzip **both** archives into **one** directory:

```text
MyColabFolder/
├── sneaker_recommendation_system.ipynb    ← from Project.zip
├── outputs/                               ← from Project.zip
└── data/                                  ← from Data.zip (must be named data/)
    ├── styles.csv
    └── images/
        └── *.jpg
```

The notebook uses:

```python
USE_LOCAL_DATA = os.path.exists("data/styles.csv") and os.path.isdir("data/images") ...
```

So `data/` must sit **in the same folder** as the `.ipynb`, not inside `Project/`.

**Recommended Colab steps:**

1. Upload `Project.zip` and `Data.zip` to Colab.
2. `!unzip Project.zip`
3. `!unzip Data.zip`
4. Open `sneaker_recommendation_system.ipynb`.
5. `Runtime → Run all`.
6. For the interview demo: uncomment `demo.launch(share=True)` in Appendix B.

---

## 5. What goes inside each ZIP

### Project.zip (code + report)

| Include | Reason |
|---------|--------|
| `sneaker_recommendation_system.ipynb` | Official report + code |
| `outputs/*.png` | Training plots (Section 7) |

| Exclude | Reason |
|---------|--------|
| `data/` | Goes in **Data.zip** |
| `checkpoints/`, `embeddings.npy` | Regenerated when the notebook runs |
| `env/`, `.git`, `Recommendation system/` | Not required |
| `ai-ml-dl.html`, quiz files | Study material only |

### Data.zip (sample only)

| Include | Reason |
|---------|--------|
| `data/styles.csv` | Kaggle metadata (notebook filters from this) |
| `data/images/*.jpg` | ~200 images (100 sneakers + 100 other) |

| Exclude | Reason |
|---------|--------|
| Full 44k image folder | Too large; not required |
| `filtered_images/`, `metadata.csv` | Created by the notebook on first run |

---

## 6. Before you submit — checklist

- [ ] Replace `<INSERT_MATRICULATION_NUMBER>` in the notebook (title + Metadata).
- [ ] Run `build_submission.py` and create `Project.zip` + `Data.zip`.
- [ ] Unzip both in one test folder locally or in Colab and run **Run all** once.
- [ ] Confirm Section 7 plots and `topk_df` table print without errors.
- [ ] AI Usage Declaration filled in honestly.
- [ ] Upload **only** `Project.zip` and `Data.zip` to Moodle (exact names).

---

## 7. Oral interview (short pitch)

> “This is a **content-based sneaker recommender**. I fine-tune **ResNet-18** with **transfer learning** on a small Kaggle fashion subset, use the 512-dim embedding layer, and rank catalog items with **cosine similarity** against a **random baseline**. Appendix B is a **Gradio demo** for live upload and shopping links; the graded part is Sections 5–7.”

---

## 8. O'zbekcha — qisqacha

1. `python3 build_submission.py` ishga tushiring.
2. `submission/` ichida `Project.zip` va `Data.zip` yarating.
3. Moodle'ga **faqat** shu ikkita faylni yuklang (nomlari aynan shunday).
4. Colab'da ikkala ZIP'ni **bir papkaga** oching: notebook yonida `data/` bo'lishi kerak.
5. `report/` papkasi — o'zingiz uchun; Moodle'ga yuborilmaydi.

Omad!

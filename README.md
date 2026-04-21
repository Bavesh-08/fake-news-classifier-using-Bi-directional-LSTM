# 📰 Fake News Detection using LSTM & Bidirectional LSTM

A deep learning project that classifies news articles as **real or fake** using Natural Language Processing (NLP) techniques and recurrent neural networks trained on the FakeNewsNet dataset.

---

## 📁 Dataset

- **Source:** `FakeNewsNet.csv`
- **Features used:** `title` (news headline)
- **Target:** `real` (1 = Real News, 0 = Fake News)
- **Total samples after cleaning:** 23,196
  - Real: 17,441
  - Fake: 5,755

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading & manipulation |
| NLTK | Stopword removal & stemming |
| TensorFlow / Keras | Model building & training |
| Scikit-learn | Train/test split, evaluation metrics |
| NumPy | Array operations |

---

## 🚀 Project Pipeline

### 1. Data Preprocessing
- Drop rows with null values
- Extract `title` as the input feature
- Clean text: remove non-alphabetic characters, convert to lowercase
- Apply **Porter Stemming** and remove **English stopwords**

### 2. Text Encoding
- **One-Hot Encoding** with vocabulary size = 10,000
- **Padding** sequences to a fixed length of 20 tokens (`pre` padding)

### 3. Model Architecture

Two models were built and compared:

#### Model 1 — Simple LSTM
```
Embedding(10000, 40) → LSTM(100) → Dense(1, sigmoid)
```

#### Model 2 — Bidirectional LSTM ✅ (Best Model)
```
Embedding(10000, 40) → Bidirectional(LSTM(100)) → Dropout(0.3) → Dense(1, sigmoid)
```

- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy

---

## 🏋️ Training

```python
model1.fit(X_train, y_train,
           validation_data=(X_test, y_test),
           epochs=10,
           batch_size=64)
```

- **Train/Test Split:** 67% / 33%
- **Random State:** 42

### Training History (Bidirectional LSTM)

| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|--------------|
| 1     | 85.89%        | 82.65%       |
| 2     | 88.29%        | 81.92%       |
| 5     | 93.58%        | 79.01%       |
| 10    | 96.90%        | 80.22%       |

> ⚠️ Note: Increasing training accuracy with decreasing validation accuracy indicates **overfitting** after epoch 1–2. Early stopping is recommended.

---

## 📊 Results

### Confusion Matrix
```
[[ 935,  919],
 [ 595, 5206]]
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fake (0) | 0.61 | 0.50 | 0.55 | 1,854 |
| Real (1) | 0.85 | 0.90 | 0.87 | 5,801 |
| **Overall Accuracy** | | | **0.80** | **7,655** |

---

## ⚠️ Limitations & Future Improvements

- The model only uses the **title** of the news article; incorporating the full body text could improve accuracy.
- **Class imbalance** (3:1 ratio of real to fake) affects performance on the fake class — consider SMOTE or class weighting.
- The current approach uses **one-hot encoding**, which is lossy; replacing it with **pre-trained embeddings** (GloVe, Word2Vec) could boost performance.
- **Overfitting** is evident — adding more Dropout layers or using **Early Stopping** is recommended.
- Consider trying **Transformer-based models** (e.g., BERT) for improved contextual understanding.

---

---

## 📌 How to Run

1. Clone the repository and open the notebook in **Google Colab** or locally.
2. Upload `FakeNewsNet.csv` to `/content/`.
3. Run all cells sequentially.
4. Install dependencies if needed:

```bash
pip install tensorflow nltk scikit-learn pandas numpy
```

---

## 📜 License

This project is for educational purposes. Dataset credits go to the FakeNewsNet benchmark repository.

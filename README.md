# mmWave Beam Selection with Deep Residual CNN + Attention

A deep learning pipeline for optimal beam selection in **mmWave MIMO radar systems**, using a custom Residual CNN with Squeeze-and-Excitation (channel attention) blocks trained on the MIMO Radar Signal Dataset from Kaggle.

---

## Overview

In millimeter-wave (mmWave) communications and radar, selecting the optimal beam from a predefined codebook is critical for maximizing signal quality. This notebook automates that selection process by:

1. Building a **DFT codebook** of steering vectors across antenna elements
2. Engineering rich features from complex-valued channel matrices
3. Training a **Deep Residual CNN with channel attention** to classify the optimal beam
4. Evaluating performance via Top-K accuracy, confusion matrices, and inference latency

---

## Dataset

**Source:** [MIMO Radar Signal Dataset](https://www.kaggle.com/datasets/shahzaibshazoo/mimo-radar-signal-dataset) via `kagglehub`

- Downloaded automatically using `kagglehub.dataset_download()`
- All `.csv` files are concatenated into a single DataFrame
- Each row contains complex-valued antenna readings (`ant_*` columns)

---

## Pipeline

### 1. Label Generation
- A **DFT codebook** with `NUM_BEAMS = 16` steering vectors is constructed across N antenna elements
- Per-sample beam gain is computed as `|H @ W*|²`
- The **optimal beam index** (argmax gain) becomes the classification label

### 2. Feature Engineering
The complex channel matrix `H` is decomposed into the following features per sample:

| Feature | Description |
|---|---|
| Real part | `H.real` |
| Imaginary part | `H.imag` |
| Magnitude | `|H|` |
| Phase (cos/sin) | Unwrapped phase decomposed into cosine and sine |
| Phase difference (cos/sin) | Inter-antenna phase differences |
| Log beam gains | `log1p(gains)` for all 16 beams |

All features are concatenated into a flat feature vector per sample.

### 3. Preprocessing
- 80/20 stratified train/test split
- `StandardScaler` fitted on training data only
- Labels one-hot encoded via `to_categorical`
- Input reshaped to `(samples, features, 1)` for 1D convolution

### 4. Model Architecture
A custom **Deep Residual CNN with Channel Attention**:

```
Input → Conv1D(64, k=7) → BN → ReLU
  → ResBlock(64) × 2 → ChannelAttention → MaxPool
  → ResBlock(128) + ResBlock(128, dilation=2) → ChannelAttention → MaxPool
  → ResBlock(256) + ResBlock(256, dilation=2) → ChannelAttention
  → GlobalAveragePooling
  → Dense(256) → Dropout(0.3)
  → Dense(128) → Dropout(0.2)
  → Dense(16, softmax)
```

**Key components:**
- **Residual blocks** with optional dilation for wider receptive fields
- **Squeeze-and-Excitation (SE) attention** for channel-wise feature recalibration
- **Cosine Decay with Restarts** LR schedule (warm restarts)
- **Label smoothing** (`0.1`) in the final training compilation

### 5. Training
- Optimizer: `Adam`
- Loss: `CategoricalCrossentropy` with label smoothing
- Callbacks: `ReduceLROnPlateau` + `EarlyStopping` (restores best weights)
- Up to 20 epochs, batch size 128, 15% validation split

---

## Evaluation

### Metrics
- **Test Accuracy** (top-1)
- **Top-K Accuracy** for K ∈ {1, 2, 3, 5}
- **Confusion Matrix** (16×16 heatmap)
- **Classification Report** (per-beam precision/recall/F1)

### Latency Analysis
- 100-sample warmup pass
- Average per-sample inference time benchmarked over 1000 samples (reported in milliseconds)

---

## Usage

### Prerequisites

```bash
pip install kagglehub tensorflow scikit-learn numpy pandas matplotlib seaborn
```

### Run the Notebook

1. Open `mmwave-improved.ipynb` in Jupyter or VS Code
2. Run all cells sequentially
3. The trained model is saved as `beam_selection_improved.keras`

### Predict on a New Sample

```python
beam_idx, probabilities = predict_beam(raw_feature_vector)
print(f"Predicted beam: {beam_idx}, Confidence: {probabilities[beam_idx]:.4f}")
```

`raw_feature_vector` should be a 1D NumPy array of raw (pre-scaling) features with the same width as the training feature matrix.

---

## File Structure

```
.
├── mmwave-improved.ipynb          # Main notebook
├── beam_selection_improved.keras  # Saved model (generated after training)
└── README.md
```

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Number of beams | 16 |
| Train/test split | 80% / 20% |
| Batch size | 128 |
| Max epochs | 20 |
| Early stopping patience | 12 |
| LR reduce patience | 5 |
| Label smoothing | 0.1 |
| Dropout (layer 1) | 0.3 |
| Dropout (layer 2) | 0.2 |

---

## References

- Hu, J. et al. — *Squeeze-and-Excitation Networks* (CVPR 2018)
- He, K. et al. — *Deep Residual Learning for Image Recognition* (CVPR 2016)
- [MIMO Radar Signal Dataset on Kaggle](https://www.kaggle.com/datasets/shahzaibshazoo/mimo-radar-signal-dataset)

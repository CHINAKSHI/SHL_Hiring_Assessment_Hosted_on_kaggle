# 🎵 SHL Hiring Assessment (Audio Feature Extraction and Grammar Score Prediction)

# 🎵 Audio Grammar Scoring

This repository contains code, analysis, and models for predicting grammar quality scores from raw `.wav` audio files using audio feature extraction techniques and machine learning.

📄 **Note:** The detailed report explaining the approach, preprocessing steps, pipeline architecture, evaluation results, and interpretation of visualizations is available in the `Report/` folder.



---


## 🔍 Approach Overview

### 🔸 Input Audio
- Raw `.wav` audio files are used as input.

### 🔸 Feature Extraction using Librosa
- MFCCs (13 coefficients)
- Chroma features
- Spectral contrast
- Zero Crossing Rate (ZCR)
- RMSE energy
- Spectral Rolloff
- Tempo

### 🔸 Feature Matrix
- All extracted features are combined into a matrix.

### 🔸 Preprocessing
- Standard scaling (mean = 0, std = 1)

### 🔸 Train/Test Split
- 80% training / 20% testing

### 🔸 Models Used
- Random Forest (balanced and unbalanced)
- XGBoost
- Gradient Boosting

### 🔸 Evaluation Metrics
- Pearson Correlation (r = 0.648)
- R² Score (0.395)
- RMSE (0.908)

---

## 🔢 Exploratory Data Analysis (EDA)

### 1. Grammar Score Distribution
- **Most Common:** 5.0  
- **Rare Scores:** 1.0, 1.5, 2.0  
- **Skew:** Right-skewed  
> 📌 Imbalance suggests use of class weighting or data augmentation.

### 2. Audio Duration
- **Majority:** < 20 seconds  
- **Outliers:** > 60 seconds  
> 📌 Outliers might affect model learning.

### 3. Sample Rate
- **Uniform:** 16 kHz  
> 📌 Ideal for speech processing.

---

## 🎼 Feature Extraction (Librosa)

| Feature           | Description                                                         |
|-------------------|---------------------------------------------------------------------|
| **MFCCs**         | Capture timbral & phonetic patterns                                 |
| **Chroma**        | Highlight harmonic content                                          |
| **Spectral Contrast** | Distinguish voiced/unvoiced transitions, articulation quality |
| **ZCR**           | Measure noisiness/disfluencies                                      |
| **RMSE**          | Quantify energy; reflects emphasis or emotion                       |
| **Spectral Rolloff** | Detects sharp consonants; spectral energy threshold             |
| **Tempo**         | Measures rhythm/fluency of speech                                   |

---

## 📊 Feature-Label Insights

### ✩ ZCR (Zero Crossing Rate)
- **Low scores** → High ZCR → Noisy/disfluent speech  
- **High scores** → Low ZCR → Clear articulation

### ✩ RMSE (Energy)
- **Low scores** → Variable RMSE → Unstable/emotional tone  
- **High scores** → Stable RMSE → Controlled delivery

### ✩ Spectral Rolloff
- **Low scores** → High rolloff → Harsh articulation  
- **High scores** → Low rolloff → Balanced speech spectrum

---

## ⚙️ Modeling Pipeline

### 🧪 Preprocessing
- Feature scaling using `StandardScaler`
- Dataset split: 80% training / 20% testing

### 🧠 Model Training & Evaluation

| Model                    | Pearson (r) | R² Score | RMSE  | MAE    | MAPE   |
|--------------------------|-------------|----------|--------|--------|--------|
| Random Forest (unbalanced) | 0.634       | 0.383    | 0.917 | ~0.75 | ~25%  |
| Random Forest (balanced)   | **0.648**   | **0.395**| 0.908 | ~0.75 | ~25%  |
| XGBoost                    | Lower performance across all metrics |

> ✅ **Best Model:** Balanced Random Forest with `GridSearchCV`

---

## 🧲 Evaluation Metric

### 📌 Pearson Correlation Coefficient (r)
- Measures linear correlation between predicted and actual grammar scores  
- **Primary evaluation metric used**

## 📁 Requirements

- **Python**: 3.x

### 📦 Libraries

```txt
numpy  
pandas  
matplotlib  
scikit-learn  
xgboost  
librosa

---

## 🏦 Dataset Structure

```bash
dataset/
├── audio_files/
│   ├── audio_1.wav
│   ├── audio_2.wav
│   └── ...
├── train.csv                 # Training data with filenames and grammar labels
├── test.csv                  # Test data with filenames and dummy labels
└── sample_submission.csv     # Submission format


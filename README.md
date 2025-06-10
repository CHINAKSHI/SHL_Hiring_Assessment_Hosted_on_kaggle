**The report required for explaining the approach, preprocessing steps, pipeline architecture, and evaluation results, and the interpretation of the visualizations, has been uploaded in the 'Report' folder.**

## SHL Hiring Assessment (Audio Feature Extraction and Grammar Score Prediction)

This project aims to predict grammar scores from raw `.wav` audio files using various audio feature extraction techniques and machine learning models.


---


## Approach

1. **Raw `.wav` Audio**
   - Input audio is in `.wav` format.

2. **Feature Extraction**
   The following audio features are extracted from the raw `.wav` files:
   - **MFCCs**: Mel-frequency cepstral coefficients (13 coefficients).
   - **Chroma Features**: Chromagram-based features representing the 12 pitch classes.
   - **Spectral Contrast**: Spectral contrast features that highlight differences between peaks and valleys in the spectrum.
   - **Zero Crossing Rate**: The rate at which the signal changes sign.
   - **RMSE Energy**: Root mean square error-based energy measurement.
   - **Spectral Rolloff**: The frequency below which a certain percentage of the total spectral energy lies.
   - **Tempo**: The speed (beats per minute) of the audio.
3. **Feature Matrix**
   - The extracted features are organized into a feature matrix.

4. **Standard Scaling**
   - The feature matrix is standardized (mean = 0, standard deviation = 1).

5. **Train/Test Split**
   - The dataset is split into training and testing sets.

6. **Model Training**
   The following models are used for training:
   - **Random Forest (Balanced)**: A random forest model with balanced class weights.
   - **XGBoost**: Extreme Gradient Boosting model.
   - **Gradient Boosting**: Gradient boosting model.

7. **Evaluation**
   The performance of the models is evaluated using the following metrics:
   - **Pearson Correlation Coefficient**: `r = 0.648`
   - **R²**: `0.395`
   - **RMSE**: `0.908`

8. **Grammar Score Prediction**
   - The trained model is used to predict grammar scores from the extracted audio features.
  
   ---

## 📦 Dataset Structure

📦 Dataset Structure

dataset/
├── audio_files/
│   ├── audio_1.wav
│   ├── audio_2.wav
│   └── ...
├── train.csv                 # Training data with filenames and grammar labels
├── test.csv                  # Test data with filenames and dummy labels
└── sample_submission.csv     # Submission format


---

## 🚀 Getting Started

To set up and run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/audio-grammar-scoring.git
cd audio-grammar-scoring

---

## 🔍 How It Works

The pipeline includes:

- Loading data and audio files  
- Exploratory Data Analysis (EDA)  
- Feature extraction using **Librosa**  
- Preprocessing and normalization  
- Model training using **Random Forest** and **XGBoost**  
- Evaluation using **Pearson correlation** and other error metrics  

---

## 📊 Exploratory Data Analysis (EDA)

### 1. Grammar Score Distribution

- **Most common score**: `5.0`  
- **Rare scores**: `1.0`, `1.5`, `2.0`  
- Distribution is **right-skewed**  

📌 _Imbalance suggests need for class weighting or data augmentation_

---

### 2. Audio Duration

- **Most clips**: `< 20 seconds`  
- **Few outliers**: `> 60 seconds`  

📌 _Long recordings may affect model performance_

---

### 3. Sample Rate

- **All clips have a uniform rate**: `16 kHz`  

📌 _Ideal for speech tasks; no need for resampling_

---

## 🎼 Feature Extraction (Librosa)

| **Feature**          | **Description**                                                     |
|----------------------|----------------------------------------------------------------------|
| `MFCCs`              | Capture timbral & phonetic patterns                                  |
| `Chroma`             | Emphasize harmonic content                                           |
| `Spectral Contrast`  | Detect voiced/unvoiced transitions & articulation quality            |
| `ZCR`                | Measures noisiness/disfluencies via signal sign changes             |
| `RMSE`               | Indicates speech energy; emotion or emphasis                         |
| `Spectral Rolloff`   | Captures sharp consonants or spectral spread                         |
| `Tempo`              | Estimates rhythm/fluency of speech                                   |

---

## 📈 Feature-Label Insights

### 🔹 ZCR (Zero Crossing Rate)
- **Low scores** → High ZCR → _Noisy or disfluent speech_  
- **High scores** → Low ZCR → _Clear articulation_

### 🔹 RMSE (Energy)
- **Low scores** → Variable RMSE → _Emotional or unstable tone_  
- **High scores** → Stable RMSE → _Smooth, controlled delivery_

### 🔹 Spectral Rolloff
- **Low scores** → High Rolloff → _Harsh articulation_  
- **High scores** → Low Rolloff → _Balanced speech_

---

## ⚙️ Modeling Pipeline

### 🧪 Preprocessing

- Features scaled using **StandardScaler**  
- Dataset split: **80% training / 20% testing**

---

### 🧠 Model Training & Results

| **Model**                 | **Pearson (r)** | **R² Score** | **RMSE** | **MAE** | **MAPE** |
|---------------------------|-----------------|--------------|----------|---------|----------|
| Random Forest (unbalanced)| 0.634           | 0.383        | 0.917    | ~0.75   | ~25%     |
| Random Forest (balanced)  | **0.648**       | **0.395**    | **0.908**| ~0.75   | ~25%     |
| XGBoost                   | Lower performance |            |          |         |          |

✅ **Best Model**: **Balanced Random Forest** with **GridSearchCV**

---

## 🧪 Evaluation Metric

### 📌 Pearson Correlation Coefficient (r)

- Used as the **primary evaluation metric**
- Measures how strongly predicted grammar scores correlate with actual scores

---




## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `librosa`
  - `pandas`
  - `matplotlib`
  
## Installation

Clone the repository and install the required libraries:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt





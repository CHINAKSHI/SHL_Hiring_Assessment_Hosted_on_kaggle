**The report required for explaining the approach, preprocessing steps, pipeline architecture, and evaluation results, and the interpretation of the visualizations, has been uploaded in the 'Report' folder.**

## SHL Hiring Assessment (Audio Feature Extraction and Grammar Score Prediction)

```🎵 Audio Grammar Scoring

This repository contains code, analysis, and models for predicting grammar quality scores from raw .wav audio files using audio feature extraction techniques and machine learning.

Note: The report explaining the approach, preprocessing steps, pipeline architecture, evaluation results, and visual interpretation is located in the Report/ folder.

🏦 Dataset Structure

dataset/
├── audio_files/
│   ├── audio_1.wav
│   ├── audio_2.wav
│   └── ...
├── train.csv                 # Training data with filenames and grammar labels
├── test.csv                  # Test data with filenames and dummy labels
└── sample_submission.csv     # Submission format

🚀 Getting Started

1. Clone the Repository

git clone https://github.com/your-username/audio-grammar-scoring.git
cd audio-grammar-scoring

2. Install Dependencies

pip install -r requirements.txt

3. Run the Main Script

python main.py

🔍 Approach Overview

Input Audio: Raw .wav audio files are used as input.

Feature Extraction: Extracted using Librosa:

MFCCs (13 coefficients)

Chroma features

Spectral contrast

Zero crossing rate

RMSE energy

Spectral rolloff

Tempo

Feature Matrix: All features are combined into a matrix.

Standard Scaling: Normalize features (mean = 0, std = 1).

Train/Test Split: 80% training, 20% testing.

Model Training:

Random Forest (balanced and unbalanced)

XGBoost

Gradient Boosting

Evaluation:

Pearson Correlation (r = 0.648)

R² Score (0.395)

RMSE (0.908)

Prediction: Trained model predicts grammar scores from audio.

🔢 Exploratory Data Analysis (EDA)

1. Grammar Score Distribution

Most common: 5.0

Rare: 1.0, 1.5, 2.0

Right-skewed distribution

📌 Imbalance suggests class weighting or data augmentation.

2. Audio Duration

Majority: < 20 seconds

Outliers: > 60 seconds

📌 Outliers might affect model learning.

3. Sample Rate

Uniform: 16 kHz

📌 Ideal for speech processing.

🎼 Feature Extraction (Librosa)

Feature

Description

MFCCs

Capture timbral & phonetic patterns

Chroma

Highlight harmonic content

Spectral Contrast

Distinguish voiced/unvoiced transitions, articulation quality

ZCR

Measure noisiness/disfluencies

RMSE

Quantify energy; reflects emphasis or emotion

Spectral Rolloff

Detects sharp consonants; spectral energy threshold

Tempo

Measures rhythm/fluency of speech

📊 Feature-Label Insights

✩ ZCR (Zero Crossing Rate)

Low scores → High ZCR → Noisy/disfluent speech

High scores → Low ZCR → Clear articulation

✩ RMSE (Energy)

Low scores → Variable RMSE → Unstable/emotional tone

High scores → Stable RMSE → Controlled delivery

✩ Spectral Rolloff

Low scores → High rolloff → Harsh articulation

High scores → Low rolloff → Balanced speech spectrum

⚙️ Modeling Pipeline

🧪 Preprocessing

Feature scaling using StandardScaler

Dataset split: 80% training / 20% testing

🧠 Model Training & Evaluation

Model

Pearson (r)

R² Score

RMSE

MAE

MAPE

Random Forest (unbalanced)

0.634

0.383

0.917

~0.75

~25%

Random Forest (balanced)

0.648

0.395

0.908

~0.75

~25%

XGBoost

Lower performance









✅ Best Model: Balanced Random Forest using GridSearchCV

🧲 Evaluation Metric

📌 Pearson Correlation Coefficient (r)

Primary evaluation metric

Measures the linear correlation between predicted and actual grammar scores

🏗️ Possible Extensions

Use deep learning models (e.g., CNN, LSTM, Wav2Vec)

Apply audio augmentation (pitch shift, noise, tempo variations)

Build ensemble models for robustness

Deploy a web interface for real-time grammar scoring

📁 Requirements

Python 3.x

Libraries:

numpy

pandas

matplotlib

scikit-learn

xgboost

librosa

👨‍💼 Author

Chinakshi ChoudharyCivil Engineer | AI Researcher | ML Enthusiast✉️ [Your Email]🌐 [LinkedIn / GitHub Profile]

📜 License

This project is licensed under the MIT License.

 ``` 

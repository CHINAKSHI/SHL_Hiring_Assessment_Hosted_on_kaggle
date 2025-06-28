import os
import numpy as np
import soundfile as sf
import librosa
import joblib
import pandas as pd
from flask import Flask, request, render_template

# Load model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_audio_features(file_path):
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = y[:, 0]

    max_duration_sec = 10
    y = y[:sr * max_duration_sec]
    duration = len(y) / sr

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1).tolist()
    mfccs_std = np.std(mfccs, axis=1).tolist()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1).tolist()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1).tolist()
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rmse = np.mean(librosa.feature.rms(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    features = (
        mfccs_mean + mfccs_std + chroma_mean +
        spectral_contrast_mean + [zcr, rmse, rolloff, duration, sr]
    )

    return pd.DataFrame([features], columns=[
        *[f"MFCC_Mean_{i+1}" for i in range(13)],
        *[f"MFCC_Std_{i+1}" for i in range(13)],
        *[f"Chroma_{i+1}" for i in range(12)],
        *[f"Spectral_Contrast_{i+1}" for i in range(7)],
        "Zero_Crossing_Rate", "RMSE", "Spectral_Rolloff", "Duration", "Sample_Rate"
    ])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return render_template('index.html', prediction_text="No file uploaded.")
    file = request.files['audio_file']
    if file.filename == '':
        return render_template('index.html', prediction_text="No file selected.")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        features_df = extract_audio_features(filepath)
        scaled = scaler.transform(features_df)
        prediction = model.predict(scaled)
        result = round(prediction[0], 2)
    except Exception as e:
        result = f"Error: {e}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return render_template('index.html', prediction_text=f"Predicted Grammar Score: {result}")

if __name__ == '__main__':
    app.run(debug=False)



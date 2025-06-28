import os
import numpy as np
import soundfile as sf
import librosa
import joblib
from flask import Flask, request, render_template

# Load model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_audio_features(file_path):
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = y[:, 0]  # mono

    # Limit to 10 seconds
    max_duration_sec = 10
    max_samples = sr * max_duration_sec
    y = y[:max_samples]

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

    return np.array(features).reshape(1, -1)


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
        features = extract_audio_features(filepath)
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)
        score = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f"Predicted Grammar Score: {score}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=False)


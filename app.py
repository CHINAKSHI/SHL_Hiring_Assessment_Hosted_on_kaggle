import os
import numpy as np
import soundfile as sf
import librosa
import joblib
from flask import Flask, request, render_template, send_from_directory

# Load trained model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Create upload directory if not exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Feature extraction function (lightweight for Render)
def extract_audio_features(file_path):
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = y[:, 0]  # Convert to mono if stereo

    # Trim to max 10 seconds to reduce memory load
    max_duration_sec = 10
    max_samples = sr * max_duration_sec
    y = y[:max_samples]

    duration = len(y) / sr

    # Extract core features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1).tolist()
    mfccs_std = np.std(mfccs, axis=1).tolist()

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1).tolist()

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1).tolist()

    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rmse = np.mean(librosa.feature.rms(y=y))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    feature_vector = (
        mfccs_mean +
        mfccs_std +
        chroma_mean +
        spectral_contrast_mean +
        [zero_crossing_rate, rmse, spectral_rolloff, duration, sr]
    )

    return np.array(feature_vector).reshape(1, -1)

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
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        output = round(prediction[0], 2)

        feature_names = (
            [f'MFCC_Mean_{i+1}' for i in range(13)] +
            [f'MFCC_Std_{i+1}' for i in range(13)] +
            [f'Chroma_{i+1}' for i in range(12)] +
            [f'Spectral_Contrast_{i+1}' for i in range(7)] +
            ['Zero_Crossing_Rate', 'RMSE', 'Spectral_Rolloff', 'Duration', 'Sample_Rate']
        )
        feature_values = features.flatten().tolist()
        feature_table = list(zip(feature_names, feature_values))

        return render_template(
            'index.html',
            prediction_text=f"Predicted Grammar Score: {output}",
            feature_table=feature_table,
            filename=file.filename
        )
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=False)  # Turn off debug mode in production


if __name__ == '__main__':
    app.run(debug=True)


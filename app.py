import numba
if hasattr(numba, "disable_jit"):
    numba.disable_jit()



from flask import send_from_directory
from flask import Flask, request, render_template
import numpy as np
import os
import librosa
import joblib

# Load trained model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Create upload directory if not exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Feature extraction function
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # MFCCs (13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1).tolist()
    mfccs_std = np.std(mfccs, axis=1).tolist()

    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1).tolist()

    # Spectral Contrast (7)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1).tolist()

    # Scalar Features (1 each)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rmse = np.mean(librosa.feature.rms(y=y))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Final 50 features = 13+13+12+7+5 scalars
    feature_vector = (
        mfccs_mean +
        mfccs_std +
        chroma_mean +
        spectral_contrast_mean +
        [zero_crossing_rate, rmse, spectral_rolloff, duration, sr]
    )

    return np.array(feature_vector).reshape(1, -1)  # Shape: (1, 50)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
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

        # Flatten feature list for display
        feature_names = [
            *['MFCC_Mean_' + str(i+1) for i in range(13)],
            *['MFCC_Std_' + str(i+1) for i in range(13)],
            *['Chroma_' + str(i+1) for i in range(12)],
            *['Spectral_Contrast_' + str(i+1) for i in range(7)],
            'Zero_Crossing_Rate',
            'RMSE',
            'Spectral_Rolloff',
            'Duration',
            'Sample_Rate'
        ]
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
    app.run(debug=True)


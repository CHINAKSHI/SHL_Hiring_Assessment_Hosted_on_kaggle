
**The file required by the report, including the process I followed and the interpretation of the visualizations, has been uploaded in the 'Report' folder.**

# Audio Feature Extraction and Grammar Score Prediction

This project aims to predict grammar scores from raw `.wav` audio files using various audio feature extraction techniques and machine learning models.

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





# Sleep-Pattern-Analysis

Website: [https://sleep-pattern-analysis.streamlit.app/](https://sleep-pattern-analysis.streamlit.app/)

## Overview

Sleep-Pattern-Analysis is a comprehensive tool designed to analyze sleep patterns, detect snoring, assess sleep quality, and identify possible sleep anomalies using both audio and physiological data. The project combines deep learning models for audio analysis, machine learning models for data classification, and an interactive Streamlit frontend for user input and results visualization.

## Features

- **Snoring Detection from Audio Files:** Uses CNN and LSTM deep learning models trained on publicly available snoring datasets to detect snoring episodes from uploaded audio recordings.
- **Sleep Anomaly & Quality Detection from CSV Data:** Employs Random Forest models to classify numerical sleep data for anomaly detection and quality assessment.
- **Personalized Recommendations:** Provides actionable sleep hygiene tips and personalized suggestions based on the analysis results.
- **Interactive Web App:** Built with Streamlit, allowing users to input data, upload audio, and view results and recommendations in a user-friendly interface.

## Setup Instructions

### 1. Audio - Snoring Detection Model

- Download the snoring dataset from [Kaggle](https://www.kaggle.com/datasets/tareqkhanemu/snoring/data).
- Train a model using CNN and LSTM architectures (see `LSTM+CNN+MFCC.py`).
- Save the trained audio model (e.g., `sleep_audio_model.h5`) in the `models` folder for the Streamlit app to access.

### 2. CSV - Sleep Factors Models

- Use two datasets:
  - One for sleep anomaly detection.
  - One for sleep quality and suggestions.
- Train Random Forest models for numerical data classification, using hyperparameter tuning and grid search for optimal accuracy.
- Save the trained models (e.g., `sleep_disorder_model.pkl`, `sleep_quality_model.pkl`) in the `models` folder.

### 3. Frontend - Streamlit App

- Ensure all required Python dependencies (see requirements below) are installed.
- Place your trained models in the `models` directory.
- Run the Streamlit app:
  ```bash
  streamlit run streamlit_app.py
  ```

## Usage

1. Open the Streamlit web application.
2. Input physiological and lifestyle data (age, gender, sleep duration, heart rate, etc.) as prompted.
3. Upload a sleep audio recording (WAV or MP3) for snoring detection.
4. Run the analysis and view detailed results, including:
   - Sleep disorder risk assessment
   - Sleep quality score
   - Snoring episode analysis and visualization
   - Personalized and general sleep recommendations

## Dataset

The snoring detection model uses the dataset created by Tareq Khan, available on Kaggle and other public sound effect repositories. The dataset consists of 500 snoring and 500 non-snoring 1-second audio clips.

If you use the dataset, please cite:
> T. H. Khan, "A deep learning model for snoring detection and vibration notification using a smart wearable gadget," Electronics, vol. 8, no. 9, article. 987, ISSN 2079-9292, 2019.

## Requirements

- Python 3.7+
- streamlit
- numpy
- pandas
- librosa
- matplotlib
- scikit-learn
- tensorflow

Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Files

- `streamlit_app.py` — Main application logic and Streamlit UI
- `LSTM+CNN+MFCC.py` — Model training script for snoring detection
- `models/` — Directory to store trained models
- `dataset/` — Contains dataset details and references

## License

_No license information provided. Please add one if needed._

## Acknowledgements

- Audio dataset courtesy of Tareq Khan, [Kaggle](https://www.kaggle.com/datasets/tareqkhanemu/snoring/data)

```

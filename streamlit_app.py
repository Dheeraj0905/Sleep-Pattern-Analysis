import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile

# Current information
CURRENT_DATE = "2025-03-07 11:00:19"
USER_LOGIN = "Mangun10"

class SleepAudioPredictor:
    def __init__(self, model_path):
        """Initialize with a pre-trained model"""
        self.model = load_model(model_path)
        self.sr = 8000  # Same as training
        self.n_mfcc = 13
    
    def extract_features(self, file_path):
        """Extract MFCC features from audio file"""
        try:
            # Load audio file
            y, _ = librosa.load(file_path, sr=self.sr, duration=1)
            
            # Ensure consistent length
            if len(y) < self.sr:
                y = np.pad(y, (0, self.sr - len(y)), 'constant')
            else:
                y = y[:self.sr]
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
            
            return mfccs
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return None
    
    def predict(self, audio_file):
        """Make prediction on uploaded audio file"""
        # Extract features
        mfccs = self.extract_features(audio_file)
        
        if mfccs is None:
            return None, None
        
        # Reshape for model input (add batch and channel dimensions)
        mfccs = mfccs[np.newaxis, ..., np.newaxis]
        
        # Make prediction
        prediction = self.model.predict(mfccs)[0]
        
        # Get result
        class_idx = np.argmax(prediction)
        confidence = float(prediction[class_idx])
        
        # Map to class name
        class_name = "Snoring" if class_idx == 1 else "Non-snoring"
        
        return class_name, confidence

def main():
    st.set_page_config(page_title="Sleep Audio Analysis", layout="wide")
    
    st.title("Sleep Audio Analysis")
    st.write(f"Current date: {CURRENT_DATE}")
    st.write(f"User: {USER_LOGIN}")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app analyzes sleep audio recordings to detect snoring patterns "
        "using a CNN-LSTM neural network model."
    )
    
    # Model path
    model_path = "models/sleep_audio_model.h5"
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Initialize predictor
    predictor = SleepAudioPredictor(model_path)
    
    # File upload
    st.subheader("Upload Audio")
    uploaded_file = st.file_uploader("Choose a sleep audio file", type=["wav"])
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format="audio/wav")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Make prediction
        with st.spinner("Analyzing audio..."):
            result, confidence = predictor.predict(temp_path)
        
        # Display results
        if result:
            st.success("Analysis complete!")
            
            # Display results in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", result)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Add explanatory text
            if result == "Snoring":
                st.info(
                    "Snoring detected in the audio. Snoring can indicate potential "
                    "sleep disorders like sleep apnea, or may be caused by nasal obstruction, "
                    "obesity, or sleeping position."
                )
            else:
                st.info(
                    "No snoring detected in the audio. This indicates normal breathing "
                    "patterns during sleep."
                )
            
            # Remove temporary file
            os.unlink(temp_path)
        else:
            st.error("Failed to process audio. Please try another file.")

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, BatchNormalization, 
                                   Flatten, Dense, LSTM, Reshape)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Current information
CURRENT_DATE = "2025-03-07 10:51:27"
USER_LOGIN = "Mangun10"

class SleepAudioProcessor:
    def __init__(self, data_path):
        """
        Initialize the sleep audio processor
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset directory containing class folders (0 and 1)
        """
        self.data_path = data_path
        self.sr = 8000  # Lower sampling rate for efficiency
        self.n_mfcc = 13  # Number of MFCCs to extract
        self.n_mels = 40  # Number of Mel bands
        
        # Create directories for model saving
        os.makedirs('models', exist_ok=True)
        
        print(f"Sleep Audio Processor initialized by {USER_LOGIN} on {CURRENT_DATE}")
        
    def extract_features(self, file_path):
        """
        Extract MFCC and Mel spectrogram features from audio file
        """
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
            
            # Extract Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mfccs, mel_spec_db
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None
    
    def load_dataset(self, max_files=None):
        """
        Load dataset and extract features
        
        Parameters:
        -----------
        max_files : int or None
            Maximum number of files to load per class (None for all)
        """
        mfccs = []
        mel_specs = []
        labels = []
        
        print(f"Loading dataset from {self.data_path}...")
        
        # Process each class directory
        for label in os.listdir(self.data_path):
            class_dir = os.path.join(self.data_path, label)
            
            if not os.path.isdir(class_dir):
                continue
                
            print(f"Processing class {label}...")
            
            # Get audio files
            audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
            
            # Limit files if specified
            if max_files:
                audio_files = audio_files[:max_files]
                print(f"Limited to {len(audio_files)} files")
                
            # Process each audio file
            for audio_file in audio_files:
                file_path = os.path.join(class_dir, audio_file)
                mfcc, mel_spec = self.extract_features(file_path)
                
                if mfcc is not None and mel_spec is not None:
                    mfccs.append(mfcc)
                    mel_specs.append(mel_spec)
                    labels.append(int(label))
        
        # Convert to numpy arrays
        X_mfcc = np.array(mfccs)
        X_mel = np.array(mel_specs)
        y = np.array(labels)
        
        # Add channel dimension for CNN
        X_mfcc = X_mfcc[..., np.newaxis]
        X_mel = X_mel[..., np.newaxis]
        
        # One-hot encode labels
        y = to_categorical(y)
        
        print(f"MFCC shape: {X_mfcc.shape}")
        print(f"Mel spectrogram shape: {X_mel.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X_mfcc, X_mel, y
    
    def build_model(self, input_shape):
        """
        Build a CNN-LSTM model for audio classification
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data
        """
        model = Sequential([
            # CNN layers
            Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Reshape for LSTM
            Reshape((-1, 32)),  # Reshape to (time_steps, features)
            
            # LSTM layer
            LSTM(64),
            Dropout(0.3),
            
            # Dense layers
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer (2 classes: snoring/non-snoring)
            Dense(2, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X, y, epochs=15, batch_size=32):
        """
        Train the model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Features (MFCC or Mel spectrogram)
        y : numpy.ndarray
            Labels
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model
        model = self.build_model(X.shape[1:])
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('models/sleep_audio_model.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        # Train model
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model.save('models/sleep_audio_model.h5')
        print("Model saved to models/sleep_audio_model.h5")
        
        return model, history
    
    def predict(self, model, audio_path):
        """
        Make prediction on audio file
        
        Parameters:
        -----------
        model : tensorflow.keras.Model
            Trained model
        audio_path : str
            Path to audio file
        """
        # Extract features
        mfcc, _ = self.extract_features(audio_path)
        
        if mfcc is None:
            return "Error processing audio file"
        
        # Reshape for model input
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        
        # Make prediction
        prediction = model.predict(mfcc)[0]
        
        # Get result
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx]
        
        # Map to class name
        class_name = "Snoring" if class_idx == 1 else "Non-snoring"
        
        return class_name, confidence


def main():
    """
    Main function
    """
    print(f"=== Sleep Audio Analysis ===")
    print(f"Date: {CURRENT_DATE}")
    print(f"User: {USER_LOGIN}")
    
    # Set path to your local dataset
    dataset_path = "dataset"  # Update this to your dataset path
    
    # Initialize processor
    processor = SleepAudioProcessor(dataset_path)
    
    # Check if model already exists
    model_path = "models/sleep_audio_model.h5"
    if os.path.exists(model_path):
        print(f"\nLoading existing model from {model_path}")
        model = load_model(model_path)
    else:
        print("\nTraining new model...")
        # Load dataset (limit to 100 files per class for faster processing)
        X_mfcc, X_mel, y = processor.load_dataset(max_files=499)
        
        # Use MFCC features for training (faster and smaller)
        model, _ = processor.train_model(X_mfcc, y, epochs=10)
    
    # Test prediction if a test file is specified
    test_file = input("\nEnter path to test audio file (or press Enter to skip): ").strip()
    if test_file and os.path.exists(test_file):
        result, confidence = processor.predict(model, test_file)
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.4f}")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
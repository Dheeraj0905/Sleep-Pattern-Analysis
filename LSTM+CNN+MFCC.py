import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, LSTM, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Current information
CURRENT_DATE = "2025-03-07 11:19:11"
USER_LOGIN = "Mangun10"

print(f"=== Sleep Audio Analysis Model Training ===")
print(f"Date: {CURRENT_DATE}")
print(f"User: {USER_LOGIN}")

def extract_features(file_path, sr=8000, n_mfcc=13):
    """Extract MFCC features from audio file"""
    try:
        # Load audio with lower sampling rate
        y, _ = librosa.load(file_path, sr=sr, duration=1)
        
        # Ensure consistent length
        if len(y) < sr:
            y = np.pad(y, (0, sr - len(y)), 'constant')
        else:
            y = y[:sr]
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def load_dataset(data_path, max_files=None):
    """Load dataset and extract features"""
    sr = 8000  # Lower sampling rate for efficiency
    n_mfcc = 13  # Number of MFCCs
    
    mfccs = []
    labels = []
    
    print(f"Loading dataset from {data_path}...")
    
    # Process each class directory (0=non-snoring, 1=snoring)
    for label in os.listdir(data_path):
        class_dir = os.path.join(data_path, label)
        
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Processing class {label}...")
        
        # Get audio files
        audio_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        
        # Limit files if specified
        if max_files:
            audio_files = audio_files[:max_files]
            print(f"Limited to {len(audio_files)} files per class")
        
        # Process each audio file
        for audio_file in audio_files:
            file_path = os.path.join(class_dir, audio_file)
            mfcc = extract_features(file_path, sr, n_mfcc)
            
            if mfcc is not None:
                mfccs.append(mfcc)
                labels.append(int(label))
    
    # Convert to numpy arrays
    X = np.array(mfccs)
    y = np.array(labels)
    
    # Add channel dimension for CNN
    X = X[..., np.newaxis]
    
    # One-hot encode labels
    y = to_categorical(y)
    
    print(f"Dataset loaded: {X.shape[0]} samples")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y

def build_model(input_shape):
    """Build a CNN-LSTM model for audio classification"""
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
    
    # Print model summary
    model.summary()
    
    return model

def train_model(X, y, epochs=15, batch_size=32):
    """Train the model"""
    # Create directory for models
    os.makedirs('models', exist_ok=True)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = build_model(X.shape[1:])
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('models/sleep_audio_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    # Train model
    print(f"\nTraining model with {epochs} epochs and batch size {batch_size}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"\nValidation accuracy: {accuracy:.4f}")
    
    # Save model
    model.save('models/sleep_audio_model.h5')
    print("Model saved to models/sleep_audio_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_history.png')
    plt.show()
    
    return model, history

# Main execution
if __name__ == "__main__":
    # Set path to dataset
    dataset_path = "C:\Users\manas\OneDrive\Desktop\projects\ML Project\Sleep-Pattern-Analysis\dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        exit(1)
    
    # Ask for training parameters
    max_files = input("Maximum files per class (press Enter for all): ")
    max_files = int(max_files) if max_files.strip() else None
    
    epochs = input("Number of training epochs (default=15): ")
    epochs = int(epochs) if epochs.strip() else 15
    
    batch_size = input("Batch size (default=32): ")
    batch_size = int(batch_size) if batch_size.strip() else 32
    
    # Load dataset
    X, y = load_dataset(dataset_path, max_files)
    
    # Train and save model
    train_model(X, y, epochs, batch_size)
    
    print("\nModel training completed!")
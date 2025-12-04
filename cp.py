import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# --- Configuration ---
# Path to your trained Cry Audio Model
MODEL_PATH = r"D:\ai-baby-care\final_baby_cry_model.keras"

# Audio processing constants (Must match training)
SAMPLE_RATE = 22050
DURATION = 5  # Seconds
TARGET_SHAPE = (224, 224)

# Classes (Alphabetical order based on LabelEncoder in your notebook)
CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

def preprocess_audio(file_path):
    """
    Loads audio, pads/truncates to 5s, converts to Mel-Spectrogram,
    resizes to 224x224, and applies ResNet preprocessing.
    """
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # 2. Pad or Truncate to exactly 5 seconds
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # 3. Generate Mel Spectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mels_db = librosa.power_to_db(mels, ref=np.max)

        # 4. Normalize to [0, 1]
        mels_db = (mels_db - mels_db.min()) / (mels_db.max() - mels_db.min() + 1e-6)

        # 5. Resize and converting to RGB (3 channels)
        # Add channel dimension for TF resizing
        mels_db = mels_db[..., np.newaxis]
        
        # Resize to (224, 224)
        resized = tf.image.resize(mels_db, TARGET_SHAPE)
        
        # Convert Grayscale to RGB (Duplicate channels)
        rgb_image = tf.image.grayscale_to_rgb(resized)

        # 6. Apply ResNetV2 specific preprocessing
        final_input = preprocess_input(rgb_image)

        # Add batch dimension (1, 224, 224, 3)
        return np.expand_dims(final_input, axis=0)

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def main():
    # Load Model
    print(f"Loading model from: {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # User Input
    audio_path = input("\nEnter path to baby cry audio file : ").strip().strip('"')
    
    if not os.path.exists(audio_path):
        print("File not found. Please check the path.")

    # Preprocess
    input_tensor = preprocess_audio(audio_path)
    
    if input_tensor is not None:
        # Predict
        predictions = model.predict(input_tensor, verbose=0)
        predicted_index = np.argmax(predictions)
        predicted_class = CLASSES[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        print(f"\n>>> Prediction: {predicted_class.upper()}")
        print(f">>> Confidence: {confidence:.2f}%")
        print("-" * 30)
        # Print all probabilities
        for i, label in enumerate(CLASSES):
            print(f"{label}: {predictions[0][i]*100:.2f}%")

if __name__ == "__main__":
    main()
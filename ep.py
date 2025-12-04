import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# --- Configuration ---
# Path to your trained Emotion Image Model
MODEL_PATH = r"D:\ai-baby-care\b3_phase2_best.keras"

# Image processing constants (Must match training)
IMG_SIZE = (256, 256)

# Classes
CLASSES = ['Angry', 'Cry', 'Laugh', 'Normal']

def preprocess_image(file_path):
    """
    Loads image, resizes to 256x256, converts to array, and batches it.
    Note: EfficientNet's 'preprocess_input' is baked into your model graph
    (based on your notebook), so we pass raw RGB values.
    """
    try:
        # 1. Load and Resize
        img = load_img(file_path, target_size=IMG_SIZE)
        
        # 2. Convert to Array
        img_array = img_to_array(img)
        
        # 3. Add batch dimension (1, 256, 256, 3)
        img_batch = tf.expand_dims(img_array, 0)
        
        return img_batch

    except Exception as e:
        print(f"Error processing image: {e}")
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
    image_path = input("\nEnter path to baby image file : ").strip().strip('"')
    
    
    if not os.path.exists(image_path):
        print("File not found. Please check the path.")

    # Preprocess
    input_tensor = preprocess_image(image_path)
    
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
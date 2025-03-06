import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image


MODEL_PATH = 'Mobilenet_Type.h5'
MODEL_PATH_2 = 'Mobilenet_Severity.h5'

TARGET_SIZE = (224, 224)

# Wound Labels
WOUND_LABELS = {
    0: 'Abrasions',
    1: 'Bruises', 
    2: 'Burns',
    3: 'Cut',
    4: 'Laseration',
    5: 'Normal'
}

WOUND_SEVERITY_LABELS = {
    0: 'Degree 1',
    1: 'Degree 2',
    2: 'Degree 3' 
}

def load_keras_model(model_path):
    """Load Keras model with error handling"""
    try:
        tf.get_logger().setLevel('ERROR')
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_image(image_file, target_size=TARGET_SIZE):
    """Preprocess the uploaded image for model prediction"""
    # Open the image
    img = Image.open(image_file)
    
    # Convert to RGB if the image is in a different mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert to array
    img_array = img_to_array(img)
    
    # Expand dimensions to create batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image
    img_array /= 255.0
    
    return img_array

def predict_wound_type(model, image_array):
    """Predict wound type using the pre-trained model"""
    # Make prediction
    predictions = model.predict(image_array)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    
    # Get the label
    label = WOUND_LABELS[predicted_class]
    
    return {
        'predicted_class': label,
        'confidence': float(confidence)
    }

def predict_wound_severity(model, image_array):
    """Predict wound type using the pre-trained model"""
    # Make prediction
    predictions = model.predict(image_array)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    
    # Get the label
    label = WOUND_SEVERITY_LABELS[predicted_class]
    
    return {
        'predicted_class': label,
        'confidence': float(confidence)
    }
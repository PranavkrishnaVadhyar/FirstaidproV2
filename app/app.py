import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from app.services.image_class import load_keras_model, preprocess_image, predict_wound_type
from app.services.rag import rag_node

# Configuration and Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'Mobilenet_TL.keras'
TARGET_SIZE = (224, 224)

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load the model globally
MODEL = load_keras_model(MODEL_PATH)
if MODEL is None:
    raise RuntimeError("Could not load the model. Please check the model file.")

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for wound type prediction"""
    # Check if file is present in the request
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file uploaded',
            'status': 'failure'
        }), 400

    file = request.files['file']

    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'error': 'No selected file',
            'status': 'failure'
        }), 400

    # Check file type
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed',
            'status': 'failure'
        }), 400

    try:
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        processed_image = preprocess_image(filepath)

        # Predict wound type
        prediction = predict_wound_type(MODEL, processed_image)

        # Remove the uploaded file
        os.remove(filepath)

        return jsonify({
            'status': 'success',
            'result': prediction
        })

    except Exception as e:
        # Log the error (in a real-world scenario, use proper logging)
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Error processing the image',
            'status': 'failure'
        }), 500


@app.route('/rag', methods=['POST'])
def rag():
    try:
        # Check if the JSON payload exists and has the 'prompt' key
        if not request.is_json:
            return jsonify({
                'error': 'Invalid input, JSON data is required',
                'status': 'failure'
            }), 400

        data = request.get_json()

        if 'prompt' not in data:
            return jsonify({
                'error': 'Prompt not given',
                'status': 'failure'
            }), 400

        prompt = data['prompt']

        # Call the `rag_node` function to get the prediction
        pred = rag_node(prompt)
        answer = pred.get('answer', 'No answer returned')

        return jsonify({
            'status': 'success',
            'result': answer
        })

    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"Prediction error: {str(e)}")

        return jsonify({
            'error': 'Error processing the request',
            'status': 'failure'
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
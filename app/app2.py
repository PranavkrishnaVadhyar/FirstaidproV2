import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_together import ChatTogether
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from pyngrok import ngrok

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'mysecret')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///site.db'  # Example with SQLite
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    QDRANT_URL='https://f29211da-7232-40f5-b19b-c647c9926e78.europe-west3-0.gcp.cloud.qdrant.io'
    QDRANT_API_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.iKNJT6ZY5DfXG80LdrzpED2hIP_Nnw8WTkdL0m97bHo'
    TOGETHER_API = 'feda905a3fc3a4250e1c91e84e8639544eb02b0c9366d4d4047d3b70e69aad92'
    
# Load Qdrant Client
embeddings = FastEmbedEmbeddings()
client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
vector_store = Qdrant(client=client, collection_name="test_rag", embeddings=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Load Together AI LLM
llm = ChatTogether(together_api_key=Config.TOGETHER_API, model="meta-llama/Llama-3-70b-chat-hf")

# Define RAG Node
def rag_node(message):
    template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Question: {input}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    result = retrieval_chain.invoke({"input": message})
    return result

# Flask App Initialization
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Models
MODEL_PATH = '/content/Mobilenet_Type.h5'
MODEL_PATH_2 = '/content/Mobilenet_Severity.h5'

def load_keras_model(model_path):
    """Load Keras model with error handling."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

MODEL = load_keras_model(MODEL_PATH)
MODEL2 = load_keras_model(MODEL_PATH_2)

if MODEL is None or MODEL2 is None:
    raise RuntimeError("Could not load models. Check the model file paths.")

# Label Mappings
WOUND_LABELS = {0: 'Abrasions', 1: 'Bruises', 2: 'Burns', 3: 'Cut', 4: 'Laceration', 5: 'Normal'}
WOUND_SEVERITY_LABELS = {0: 'Degree 1', 1: 'Degree 2', 2: 'Degree 3'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_file, target_size=(224, 224)):
    """Preprocess the uploaded image for model prediction."""
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_wound_type(model, image_array):
    """Predict wound type using the pre-trained model."""
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    return {'predicted_class': WOUND_LABELS[predicted_class], 'confidence': float(confidence)}

def predict_wound_severity(model, image_array):
    """Predict wound severity using the pre-trained model."""
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    return {'predicted_class': WOUND_SEVERITY_LABELS[predicted_class], 'confidence': float(confidence)}

# Prediction API - Wound Type
@app.route('/predict_type', methods=['POST'])
def predict_type():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'status': 'failure'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG allowed', 'status': 'failure'}), 400

    try:
        processed_image = preprocess_image(file)
        prediction = predict_wound_type(MODEL, processed_image)
        return jsonify({'status': 'success', 'wound_type': prediction})
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Error processing the image', 'status': 'failure'}), 500

# Prediction API - Wound Severity
@app.route('/predict_severity', methods=['POST'])
def predict_severity():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'status': 'failure'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG allowed', 'status': 'failure'}), 400

    try:
        processed_image = preprocess_image(file)
        prediction = predict_wound_severity(MODEL2, processed_image)
        return jsonify({'status': 'success', 'wound_severity': prediction})
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Error processing the image', 'status': 'failure'}), 500

# RAG Endpoint
@app.route('/rag', methods=['POST'])
def rag():
    try:
        data = request.get_json()
        if 'prompt' not in data:
            return jsonify({'error': 'Missing prompt', 'status': 'failure'}), 400
        response = rag_node(data['prompt'])
        return jsonify({'status': 'success', 'answer': response['answer']})
    except Exception as e:
        print(f"RAG error: {str(e)}")
        return jsonify({'error': 'Error processing the query', 'status': 'failure'}), 500
# Start ngrok
port = 5000
ngrok.set_auth_token('2tx1jxJoQ2MQgzqcCtgTNU88VzI_VSopNY3JGzSdRzXw7dWG')
public_url = ngrok.connect(port).public_url
print(f"Ngrok URL: {public_url}")

if __name__ == '__main__':
    app.run(debug=True, port=port)

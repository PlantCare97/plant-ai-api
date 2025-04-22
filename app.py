from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import google.generativeai as genai
import io

app = Flask(__name__)

model_path = "vgg16_model.h5"
if not os.path.exists(model_path):
    import gdown
    gdown.download("https://drive.google.com/uc?id=1-AeiQr_5LfyCBvk8VqKaBW3-mtNlvZtK", model_path, quiet=False)

loaded_model_VGG16 = load_model(model_path)

# Gemini setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

class_names = ['Apple', 'Blueberry', 'Corn_(maize)', 'Grape']

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img_pil = Image.open(file.stream)

    try:
        response = model.generate_content([
            "Is this image a plant? Respond with only 'Yes' or 'No'.",
            img_pil
        ])
        is_plant = response.text.strip().lower() == 'yes'

        if is_plant:
            img = img_pil.resize((200, 200))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            predictions = loaded_model_VGG16.predict(img_array)
            predicted_class = np.argmax(predictions, axis=-1)[0]
            predicted_class_name = class_names[predicted_class]
        else:
            predicted_class_name = "Unknown"

        return jsonify({'predicted_class': predicted_class_name}), 200

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

# Vercel expects this handler
def handler(environ, start_response):
    return app(environ, start_response)

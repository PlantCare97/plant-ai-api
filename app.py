
import os
from flask import Flask, request, jsonify
import google.generativeai as genai
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model

# Download model from Google Drive if not already present
model_path = "vgg16_model.h5"
if not os.path.exists(model_path):
    gdown.download("https://drive.google.com/uc?id=YOUR_FILE_ID", model_path, quiet=False)

# Load the model
loaded_model_VGG16 = load_model(model_path)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

class_names = ['Apple', 'Blueberry', 'Corn_(maize)', 'Grape']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img_path = os.path.join('/tmp', file.filename)
    file.save(img_path)

    img_pil = Image.open(img_path)

    try:
        response = model.generate_content([
            "Is this image a plant? Respond with only 'Yes' or 'No'.",
            img_pil
        ])
        is_plant = response.text.strip().lower() == 'yes'

        if is_plant:
            img = image.load_img(img_path, target_size=(200, 200))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            predictions = loaded_model_VGG16.predict(img_array)
            predicted_class = np.argmax(predictions, axis=-1)[0]
            predicted_class_name = class_names[predicted_class]
        else:
            predicted_class_name = "Unknown"

        os.remove(img_path)
        return jsonify({'predicted_class': predicted_class_name}), 200

    except Exception as e:
        os.remove(img_path)
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

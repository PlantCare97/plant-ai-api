import os
import numpy as np
from PIL import Image
import gdown
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import google.generativeai as genai

# Load Gemini API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model_gemini = genai.GenerativeModel("gemini-2.0-flash")

# Assuming the model is placed in the root of the repo
model_path = "vgg16_model.keras"

# Load the Keras model directly from the repo
loaded_model = load_model(model_path)
print("Model loaded successfully:", loaded_model.summary())

# Class names
class_names = ['Apple_Black-rot', 'Apple_healthy', 'Cherry_Powdery-mildew', 'Cherry_healthy',
               'Corn_Northern-Leaf-Blight', 'Corn_healthy', 'Grape_Black-rot', 'Grape_healthy']


def handler(request):
    if request.method != "POST":
        return {"statusCode": 405, "body": "Only POST method allowed"}

    form = request.files
    if "file" not in form:
        return {"statusCode": 400, "body": "No file uploaded"}

    file = form["file"]
    img = Image.open(BytesIO(file.read()))

    try:
        # Gemini: Is it a plant?
        response = model_gemini.generate_content([
            "Is this image a plant? Respond with only 'Yes' or 'No'.",
            img
        ])
        is_plant = response.text.strip().lower() == 'yes'

        if is_plant:
            # Prepare image for model
            img_resized = img.resize((200, 200))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = loaded_model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=-1)[0]
            class_name = class_names[predicted_class]
        else:
            class_name = "Unknown"

        return {
            "statusCode": 200,
            "body": class_name
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }

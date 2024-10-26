from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io
import os

app = Flask(__name__)

# Configuration
MODEL_ID = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"
USE_GPU = torch.cuda.is_available() and app.config.get("ENV") == "production"

# Load the model and tokenizer
device = torch.device("cuda" if USE_GPU else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, revision=MODEL_REVISION
).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/describe', methods=['POST'])
def describe_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Encode image
    enc_image = model.encode_image(image).to(device)
    
    # Generate answer
    answer = model.answer_question(enc_image, "Describe this image.", tokenizer)
    
    return jsonify({"description": answer}), 200

if __name__ == '__main__':
    # Set debug mode for development
    app.run(debug=app.config.get("ENV") != "production", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

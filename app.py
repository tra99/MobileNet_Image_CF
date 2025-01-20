from flask import Flask, request, render_template, jsonify
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from werkzeug.utils import secure_filename
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/predicted'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = './mobilenet_v2.pth'
CLASS_NAMES = ['airplane', 'bicycles', 'cars', 'motorbikes', 'ships']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load MobileNet model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
except Exception as e:
    logger.error(f"Failed to load MobileNet model: {str(e)}")
    raise

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Predict the class of an image using MobileNet."""
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_class_idx = outputs.max(1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0, predicted_class_idx].item()

        predicted_class = CLASS_NAMES[predicted_class_idx.item()]
        return {
            "class": predicted_class,
            "confidence": confidence,
            "output_image": f"/{image_path}"
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if 'file' in request.files:
            return handle_file_upload(request.files['file'])
        elif 'path' in request.form:
            return handle_path_input(request.form['path'])
        else:
            return jsonify({"error": "No file or path provided"}), 400
    except Exception as e:
        logger.error(f"Prediction request error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def handle_file_upload(file):
    """Handle file upload for prediction."""
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict_image(filepath)
    return jsonify(result)

def handle_path_input(image_path):
    """Handle local file path for prediction."""
    if not os.path.exists(image_path):
        return jsonify({"error": "File path does not exist"}), 400

    result = predict_image(image_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

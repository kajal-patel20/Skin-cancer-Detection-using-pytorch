from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Grad-CAM implementation


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # Register hooks
        self._register_hooks(target_layer)

    def _register_hooks(self, layer):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        forward_handle = layer.register_forward_hook(forward_hook)
        backward_handle = layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([forward_handle, backward_handle])

    def __call__(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)

        # Get gradients and activations
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        # Calculate weights
        weights = np.mean(gradients, axis=(1, 2))

        # Calculate heatmap
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()

# Skin Cancer Classifier Model


class SkinCancerClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = resnet50(weights=None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Initialize model and load weights


def load_model():
    checkpoint = torch.load(
        'checkpoints/ham10000_epoch_10.pth', map_location=torch.device('cpu'))
    model = SkinCancerClassifier(num_classes=len(checkpoint['class_to_idx']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get the target layer for Grad-CAM (last convolutional layer)
    target_layer = model.base_model.layer4[-1].conv3
    return model, checkpoint['class_to_idx'], target_layer


model, class_to_idx, target_layer = load_model()
idx_to_class = {v: k for k, v in class_to_idx.items()}
grad_cam = GradCAM(model, target_layer)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_heatmap(image, heatmap):
    # Convert image to numpy array
    img = np.array(image.resize((224, 224)))

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap to heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap on original image
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    superimposed_img = Image.fromarray(superimposed_img)

    # Convert to base64 for web display
    buffered = BytesIO()
    superimposed_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Process image
            original_image = Image.open(filepath).convert('RGB')
            input_tensor = transform(original_image).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = idx_to_class[predicted.item()]
                confidence = torch.nn.functional.softmax(outputs, dim=1)[
                    0] * 100
                confidence = round(confidence[predicted.item()].item(), 2)

            # Generate Grad-CAM heatmap
            heatmap = grad_cam(input_tensor, predicted)
            heatmap_img = generate_heatmap(original_image, heatmap)

            # Clean up
            os.remove(filepath)

            return jsonify({
                'prediction': prediction,
                'confidence': confidence,
                'heatmap': heatmap_img
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    app.run(debug=True)

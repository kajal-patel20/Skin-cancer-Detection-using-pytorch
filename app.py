import os
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Classes
class_names = ['benign', 'malignant']

# Load model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(128, len(class_names))
)
model.load_state_dict(torch.load("model.pth", map_location='cpu'))
model.eval()

# Hook for Grad-CAM
features = None


def forward_hook(module, input, output):
    global features
    features = output


model.layer4.register_forward_hook(forward_hook)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def generate_heatmap(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # Forward pass
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    pred_class = output.argmax(dim=1).item()
    pred_prob = probs[0, pred_class].item() * 100  # Confidence percentage

    # Grad-CAM
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    gradients = model.layer4[1].conv2.weight.grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    cam = torch.zeros(features.shape[2:], dtype=torch.float32)
    for i in range(features.shape[1]):
        cam += pooled_gradients[i] * features[0, i, :, :]

    cam = np.maximum(cam.detach().numpy(), 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    orig = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(orig, 0.6, cam, 0.4, 0)

    heatmap_path = os.path.join(
        app.config['RESULT_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(heatmap_path, superimposed_img)

    return class_names[pred_class], pred_prob, heatmap_path


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        prediction, pred_prob, heatmap_path = generate_heatmap(filepath)
        return render_template('index.html', original=filepath, heatmap=heatmap_path, prediction=prediction, confidence=pred_prob)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

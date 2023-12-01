from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTImageProcessor, ViTForImageClassification # ViTFeatureExtractor
from torchvision.models import VisionTransformer
from PIL import Image
import warnings
import joblib
warnings.filterwarnings('ignore')
import os
from datetime import datetime



app = Flask(__name__)

# Front Page with Model 
@app.route('/')
def front_page():
    return render_template('home.html')

# Putting image in ViT model 
@app.route('/vit_model', methods=['GET', 'POST'])
def vit_model():
    return render_template('ViT_Model.html')

# Getting prediction of ViT image model 
@app.route('/vitpred', methods=['GET', 'POST'])
def outcome_page():
    uploaded_image = request.files['image']
    input_image = Image.open(uploaded_image).convert('RGB')
    model = torch.load('vitmodel.pth')
    
    # Define transformations for the training and validation sets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming RGB images
    ])
    # Preprocess the image
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
    
    # Make predictions
    with torch.no_grad():
        output = model(input_tensor)

    # Get predicted class
    probabilities = torch.nn.functional.softmax(output.logits[0])
    predicted_class = torch.argmax(output[0]).item()
    probability = float(probabilities[predicted_class])
    # Mapping from numeric class labels to names
    class_mapping = {
        0: 'Amoeba',
        1: 'Euglena',
        2: 'Hydra',
        3: 'Paramecium',
        4: 'Rod bacteria',
        5: 'Spherical bacteria',
        6: 'Spiral bacteria',
        7: 'Yeast'
    }    
    # Use the mapping to get the human-readable class name
    predicted_class_name = class_mapping.get(predicted_class, f'Unknown Class {predicted_class}')
    # Save the processed image to a file
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    filename = os.path.join(upload_folder, 'image.jpg')
    input_image.save(filename)

    return render_template('vitpred.html', prediction = predicted_class_name, probability = probability, image_path = filename)


if __name__ == '__main__':
    app.run(debug=False)
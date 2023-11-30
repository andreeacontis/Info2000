from flask import Flask, render_template, request, redirect, url_for
from transformers import ViTImageProcessor, ViTForImageClassification # ViTFeatureExtractor
from PIL import Image
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime


app = Flask(__name__)

# Front Page with Models to choose from 
@app.route('/')
def front_page():
    return render_template('home.html')

# Model Selection Page Redirection for Model Choosen 
@app.route('/model', methods=['GET', 'POST'])
def model():
    model = request.form['model']
    if model == 'ViT Model':
        return redirect(url_for('vit_model'))
    elif model == 'Yolo Model':
        return redirect(url_for('yolo_model'))
 
    return render_template('home.html')  # Redirect to choose model page if model is not recognized

# Putting image in ViT model 
@app.route('/vit_model', methods=['GET', 'POST'])
def vit_model():
    return render_template('ViT_Model.html')

# Putting image in Yolo model 
@app.route('/yolo_model', methods=['GET', 'POST'])
def yolo_model():
    return render_template('Yolo_Model.html')

# Getting prediction of ViT image model 
@app.route('/vitpred', methods=['GET', 'POST'])
def outcome_page():
    uploaded_image = request.files['image']
    image = Image.open(uploaded_image)
    
    # Instantiate the feature extractor specific to the model checkpoint
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # Instantiate the pretrained model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Extract features (patches) from the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Predict by feeding the model (** is a python operator which unpacks the inputs)
    outputs = model(**inputs)

    # Convert outputs to logis
    logits = outputs.logits

    # model predicts one of the classes by pick the logit which has the highest probability
    predicted_class_idx = logits.argmax(-1).item()

    prediction =  model.config.id2label[predicted_class_idx]
    # Save the processed image to a file
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    filename = os.path.join(upload_folder, 'image.jpg')
    image.save(filename)
    
    return render_template('vitpred.html', prediction = prediction, image_path=filename)

# Getting prediction of Yolo image model 
@app.route('/yolopred', methods=['GET', 'POST'])
def yolopred():
    uploaded_image = request.files['image']
    image = Image.open(uploaded_image)
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    # Run inference on image
    results = model(image, verbose=False)  # results list

    # Save the processed image with filename
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    image_path = os.path.join(upload_folder, f'results_{timestamp}.png')
        
    image_path = os.path.join(upload_folder, f'results.png')
    
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save(image_path)  # save image

    # Get predicted classes
    predclass = []
    for result in results:                                         # iterate results
        boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
        for box in boxes:                                          # iterate boxes
            predclass.append(result.names[int(box.cls[0])])      # Get predicted classes
    predclass = predclass
    
    return render_template('yolopred.html', image_path=image_path, predclass=predclass)

if __name__ == '__main__':
    app.run(debug=False)
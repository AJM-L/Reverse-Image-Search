# app.py
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from PIL import Image
from torchvision import transforms
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from resources import cosine_similarity, Img2VecResnet18

# Initialize Flask App and Folders
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads/'
IMAGE_FOLDER = 'images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Start app by rendering upload.html
@app.route('/')
def upload_form():
    """
    Route to display the upload form and list images from the images folder.
    """
    images = random.sample(list_images_with_folders(app.config['IMAGE_FOLDER']), 50)
    return render_template('upload.html', images=images, image_folder=app.config['IMAGE_FOLDER'])

# After receiving an upload, render result.html
@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Route to handle image uploads and perform reverse image search and classification.
    """
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        result = reverse_image_search(filepath)
        classification = image_classification(filepath)
        return render_template('result.html', result=result, classification=classification, image_path=filepath, image_folder=app.config['IMAGE_FOLDER'], upload_folder=app.config["UPLOAD_FOLDER"])

# Route for displaying images from the images folder
@app.route('/images/<path:filename>')
def image_file(filename):
    """
    Route to display images from the images folder.
    """
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

# Route for displaying uploaded images
@app.route('/uploads/<path:filename>')
def upload_file(filename):
    """
    Route to display images from the uploads folder.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Network used for artist classification
class Net(nn.Module):
    def __init__(self):
        """
        Initialize the neural network layers.
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 10 * 10, 50)

    def forward(self, input):
        """
        Define the forward pass of the network.
        """
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 10 * 10)
        output = self.fc1(output)
        return output

def list_images_with_folders(folder):
    """
    List images in the given folder and its subfolders with folder names.

    Args:
        folder (str): Path to the folder.

    Returns:
        list: List of dictionaries with each image's path and folder.
    """
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif')):
                images.append({
                    'path': os.path.join(root, file),
                    'folder': os.path.basename(root)
                })
    return images

def reverse_image_search(image_path, number_results=12):
    """
    Perform reverse image search and return the closest images.

    Args:
        image_path (str): Path to the uploaded image.
        number_results (int): Number of results to return.

    Returns:
        list: List of dictionaries with the paths and folders of each image.
    """
    inputDim = (224, 224)
    transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])
    allVectors = np.load('vectors.npz')
    
    with open('paths.csv') as csv_file:
        reader = csv.reader(csv_file)
        paths = dict(reader)

    I = Image.open(image_path).convert('RGB')
    I = transformationForCNNInput(I)
    img2vec = Img2VecResnet18()
    vec = img2vec.getVec(I)
    I.close()

    similarity = list(allVectors.keys())
    similarity.sort(key=lambda x: cosine_similarity(vec, allVectors[x]), reverse=True)
    result_paths = []
    for image in similarity[:number_results]:
        result_paths.append({"path": os.path.join(paths[image], image), "folder": str(paths[image].split("/")[-1])})
    return result_paths

def image_classification(image_path):
    """
    Classify the uploaded image and return the predicted class.

    Args:
        image_path (str): Path to the uploaded image.

    Returns:
        str: Predicted class.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    classes = tuple()
    for root, dirs, files in os.walk("images"):
        classes += tuple(sorted(dirs))
    
    I = Image.open(image_path).convert('RGB')
    I = transform(I)
    I = I.unsqueeze(0)  # Add batch dimension, shape: [1, 3, 32, 32]
    
    model = Net()
    model.load_state_dict(torch.load("./cifar_net.pth"))
    model.eval()
    
    with torch.no_grad():
        out = model(I)
        _, predicted = torch.max(out, 1)
    
    return classes[predicted.item()]

# Run app
if __name__ == "__main__":
    app.run(debug=True)

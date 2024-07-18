# app.py
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from PIL import Image
from torchvision import transforms, datasets
from resources import cosine_similarity, Img2VecResnet18
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads/'
IMAGE_FOLDER = 'images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def list_images_with_folders(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif')):
                images.append({
                    'path': os.path.join(root, file),
                    'folder': os.path.basename(root)
                })
    return images

@app.route('/')
def upload_form():
    # List images in the 'images' folder and its subfolders with folder names
    images = random.sample(list_images_with_folders(app.config['IMAGE_FOLDER']), 50)
    return render_template('upload.html', images=images, image_folder=app.config['IMAGE_FOLDER'])

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # Call your reverse image search function here
        result = reverse_image_search(filepath)
        classification = image_classification(filepath)
        return render_template('result.html', result=result, classification= classification, image_path=filepath, image_folder=app.config['IMAGE_FOLDER'])

@app.route('/images/<path:filename>')
def image_file(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 50)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output

def reverse_image_search(image_path, number_results = 10):
    inputDim = (224,224)
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

    #make a list of images
    similarity=list(allVectors.keys())
    #sort list by similarity to query
    similarity.sort(key = lambda x: cosine_similarity(vec, allVectors[x]), reverse = True)
    result_paths = []
    for image in similarity[:number_results]:
        result_paths.append({"path":(os.path.join(paths[image], image)), "folder" : str(paths[image].split("/")[-1])})
        print(image, cosine_similarity(vec, allVectors[image]))
    return result_paths

def image_classification (image_path):
    transform = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    classes = tuple()
    for root, dirs, files in os.walk("images"):
        classes += tuple(sorted(dirs))
    
    I = Image.open(image_path).convert('RGB')
    I = transform(I)
    
    # Add batch dimension
    I = I.unsqueeze(0)  # Shape: [1, 3, 32, 32]
    
    # Initialize the model and load the trained weights
    model = Net()
    model.load_state_dict(torch.load("./cifar_net.pth"))
    model.eval()
    
    # Predict the class
    with torch.no_grad():
        out = model(I)
        _, predicted = torch.max(out, 1)
    
    return classes[predicted.item()]



if __name__ == "__main__":
    app.run(debug=True)

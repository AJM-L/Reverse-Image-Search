# app.py
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from PIL import Image
from torchvision import transforms
from resources import cosine_similarity, Img2VecResnet18
import csv
import os
import numpy as np

app = Flask(__name__)
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
    images = list_images_with_folders(app.config['IMAGE_FOLDER'])
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
        return render_template('result.html', result=result, image_folder=app.config['IMAGE_FOLDER'])

@app.route('/images/<path:filename>')
def image_file(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

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
    print(similarity)
    for image in similarity[:number_results]:
        result_paths.append({"path":(os.path.join(paths[image], image)), "folder" : str(paths[image].split("/")[-1])})
        print(image, cosine_similarity(vec, allVectors[image]))
    return result_paths

if __name__ == "__main__":
    app.run(debug=True)

<h1>Reverse Image Search for Painting Reccomendation</h1>

<p>
Have you ever found a painting that you love, but have no idea how to find similar works, or created something yourself and wondered what other artists have made something similar? This project attempts to tackle these questions by giving users a method of searching a dataset of paintings using only an image as input.
  
  This project uses a kaggle <a href = "https://www.kaggle.com/datasets/mexwell/famous-paintings">dataset of famous paintings</a> and cosine similarity to retrieve images from the dataset that are similar to the input. It includes a jupyter notebook for file setup, and uses flask and html for the frontend. Re-run the jupyter notebook if you decide to add photos to the dataset, run app.py if you want to use your own image(s) as queries. 
</p>

<h2>Running the App</h2>

<p>
install required libraries

run app.py
  
</p>


<h2>Adding images</h2>

<p>
install required libraries

Add images to images folder. Make sure images have unique names and are placed in their perspective artist folder. If you decide to add more artists follow the format of Artist_Name using underscores instead of spaces. 

Run ReverseImageSearch.ipynb to create the new vectors.

Run ArtistClassification.ipynb to retrain the classifier

Finally Run app.py.
  
</p>

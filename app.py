import os
import matplotlib.pyplot as plt

from hackathon.helpers.dataset_helpers import load_data

if __name__ == "__main__":
    path = "dataset"
    datadir = os.path.join(path, "african_plums")
    print("[INFO] loading datasets...")
    data, labels = load_data(datadir)
    print("[INFO] loaded datasets...")

    # Affiche la première image du dataset
    img = data[0]  # Directement la matrice d'image déjà chargée
    
    plt.figure(figsize=(20, 10))
    plt.imshow(img)  # matplotlib attend un tableau d'image
    plt.axis('off')
    plt.show()
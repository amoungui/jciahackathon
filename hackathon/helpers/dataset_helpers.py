import sys
import os

import numpy as np
from PIL import Image

def load_data(dataset_dir):
    images = []
    labels = []
    label_map = {
        "bruised": 0,
        "cracked": 1,
        "rotten": 2,
        "spotted": 3,
        "unaffected": 4,
        "unripe": 5
    }

    for category, label in label_map.items():
        category_dir = os.path.join(dataset_dir, category)
        print(f"Traitement du dossier : {category_dir}")
        
        if os.path.exists(category_dir):
            for image_file in os.listdir(category_dir):
                print(f"Chargement de l'image : {image_file}")
                image_path = os.path.join(category_dir, image_file)

                try:
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize((32, 32))
                    image = np.array(image, dtype=np.float32) / 255.0

                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {image_path}: {e}")
        else:
            print(f"Dossier introuvable : {category_dir}")

    print(f"Total images chargées : {len(images)}")
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    pass
    # print(sys.path)

    # path = "dataset"
#    datadir = os.path.join(path, "african_plums")
#    print("[INFO] loading datasets...")
#    data, labels = load_data(datadir)
#    print("[INFO] loaded datasets...")
#    print(data[0])
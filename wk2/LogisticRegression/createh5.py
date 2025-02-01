import numpy as np
import os
import copy
import matplotlib.pyplot as plt #to plot graphs in python
import h5py #common package to interact with a dataset that is stored on an H5 file
import scipy
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split

# NOT NEEDED/USED, GOOD PRACTICE TO TRY TURNING IMAFES INTO H5 FILES/DATASET
# NOTE: TO USE WOULD HAVE TO COME AND CHANGE THE FILE PATHS, OBSELETE FILE AND THE DATA DOES NOT EXIST THERE ANYMORE 
def preProcessData():
    # PRE PROCESS THE DATASET
    # Set dataset path
    # Get the directory of the current script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Define the relative path from this script's directory
    DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "kagglecatsanddogs_5340", "PetImages")
    # DATASET_PATH = "../LogisticRegression/Dataset/kagglecatsanddogs_5340/PetImages"
    CATEGORIES = ["Cat", "Dog"]  # label Cat as 1, Dog as 0
    IMAGE_SIZE = (64, 64)  # Resize images to 64x64 pixels

    # Lists to store images and labels
    images = []
    labels = []

    # Load and preprocess images, first go through all cats then all dogs
    for category in CATEGORIES:
        folder_path = os.path.join(DATASET_PATH, category)
        # Iterate through all images in the folder
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert("RGB")  # Ensure 3 color channels
                img = img.resize(IMAGE_SIZE)
                img_array = np.array(img)
                images.append(img_array)

                # Assign labels: Cat = 1, Dog = 0
                label = 1 if category == "Cat" else 0
                labels.append(label)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")  # Handle errors like unreadable images

    # Convert to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Train-test split (80% train, 20% test)
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=42)


    # Save dataset in .h5 format
    with h5py.File("cats_vs_dogs.h5", "w") as hf:
        hf.create_dataset("train_x", data=train_x)
        hf.create_dataset("train_y", data=train_y)
        hf.create_dataset("test_x", data=test_x)
        hf.create_dataset("test_y", data=test_y)

    print("Dataset saved as cats_vs_dogs.h5 with labels: Cat=1, Non-cat(Dog)=0")

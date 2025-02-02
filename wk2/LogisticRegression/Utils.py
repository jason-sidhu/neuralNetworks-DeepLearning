import numpy as np
import os
import h5py #common package to interact with a dataset that is stored on an H5 file


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_dataset():
    trainFile = os.path.join(BASE_DIR, "Dataset", "kagglecatsanddogs_5340", "train_catvnoncat.h5")  
    testFile = os.path.join(BASE_DIR, "Dataset", "kagglecatsanddogs_5340", "test_catvnoncat.h5")  
    train_dataset = h5py.File(trainFile, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(testFile, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
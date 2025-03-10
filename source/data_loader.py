import pickle
import glob
import os
import numpy as np

TRAIN = "data/training/*.p"
VALIDATION = "data/validation/*.p"
TEST = "data/test/*.p"

def get_label_string(filepath):
    file = os.path.basename(filepath)
    return file.split('__')[0]

def generate_label_map(paths):
    all_files = []
    for path in paths:
        all_files.extend([os.path.basename(file) for file in glob.glob(path)])
    files = [get_label_string(file) for file in all_files]
    labels = set(files)

    label_map = {}
    for idx, label in enumerate(labels):
        label_map[label] = idx

    return label_map

def load_data(training_path=TRAIN, validation_path=VALIDATION, test_path=TEST):
    label_map = generate_label_map([training_path, validation_path, test_path])
    n_classes = len(label_map.keys())

    files = glob.glob(training_path)
    Xtrain = np.array([pickle.load(open(file, "rb" )) for file in files])
    ytrain = np.array([label_map[get_label_string(file)] for file in files])
    assert Xtrain.shape[0] == ytrain.shape[0], "Observation count does not match label count in training set"

    files = glob.glob(validation_path)
    Xval = np.array([pickle.load(open(file, "rb" )) for file in files])
    yval = np.array([label_map[get_label_string(file)] for file in files])
    assert Xval.shape[0] == yval.shape[0], "Observation count does not match label count in validation set"

    files = glob.glob(test_path)
    Xtest = np.array([pickle.load(open(file, "rb" )) for file in files])
    ytest = np.array([label_map[get_label_string(file)] for file in files])
    assert Xtest.shape[0] == ytest.shape[0], "Observation count does not match label count in test set"

    return Xtrain, ytrain, Xval, yval, Xtest, ytest, n_classes
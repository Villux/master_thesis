import pickle
import glob
import os
import numpy as np

def move_file_to_folder(old_path, new_path):
    os.rename(old_path, new_path)

def split_data(train_perc, val_perc, test_perc):
    assert train_perc + val_perc + test_perc == 1, "Whole dataset not used"
    files = [os.path.basename(path) for path in glob.glob("data/*.p")]

    N = len(files)
    train_size = int(N * train_perc)
    val_size = int(N * val_perc)

    np.random.shuffle(files)
    training_set, validation_set, test_set = files[:train_size], files[train_size:(train_size+val_size)], files[(train_size+val_size):]
    for file in training_set:
        move_file_to_folder(f"data/{file}", f"data/training/{file}")
    for file in validation_set:
        move_file_to_folder(f"data/{file}", f"data/validation/{file}")
    for file in test_set:
        move_file_to_folder(f"data/{file}", f"data/test/{file}")

    return "data/training/*.p", "data/validation/*.p", "data/test/*.p"

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

def load_data(training_path, validation_path, test_path):
    label_map = generate_label_map([training_path, validation_path, test_path])

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

    return Xtrain, ytrain, Xval, yval, Xtest, ytest
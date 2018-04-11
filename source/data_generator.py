import pickle
import glob
import os
import numpy as np

from heston import heston_dynamic_milstein_scheme

def generate_sample(kappa, theta, xi, rho, dt, T):
    parameters = {
        'kappa': kappa,
        'theta': theta,
        'xi': xi,
        'rho': rho
    }
    return heston_dynamic_milstein_scheme(parameters=parameters, T=T, dt=dt)

def store_sample(returns, variances, filename):
    data = np.stack((returns, variances))
    pickle.dump(data, open(f"data/{filename}.p", "wb"))

def generate_data(parameters, number_of_samples=10, dt=0, T=0):
    for _, kappa in enumerate(parameters['kappa']):
        for _, theta in enumerate(parameters['theta']):
            for _, rho in enumerate(parameters['rho']):
                for _, xi in enumerate(parameters['xi']):
                    for idx in range(number_of_samples):
                        r, v, _ = generate_sample(kappa, theta, xi, rho, dt, T)
                        filename = f"k{kappa}_t{theta}_xi{xi}_rho{rho}__{idx}"
                        store_sample(r, v, filename)

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

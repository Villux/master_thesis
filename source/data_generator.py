import pickle
import glob
import os
import h5py
import numpy as np
from multiprocessing import Pool

from heston import heston_dynamic_milstein_scheme

class SimulationEnginer(object):
    def __init__(self, kappa, theta, xi, rho, dt, T):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt
        self.T = T
    def __call__(self, idx):
        r, v, _ = generate_sample(self.kappa, self.theta, self.xi, self.rho, self.dt, self.T)
        filename = f"k{self.kappa}_t{self.theta}_xi{self.xi}_rho{self.rho}__{idx}"

        return r, v, filename


class FileWriter(object):
    def __init__(self, dataset_name, chunk):
        path = f"data/{dataset_name}.h5"
        self.file = h5py.File(path, 'w')

        N, H, W = chunk.shape
        self.row_count = 0
        max_shape = (None, H, W) + N
        self.dset = self.file.create_dataset('data', shape=(0, H, W), maxshape=max_shape,
                                chunks=chunk.shape, dtype=chunk.dtype)

    def write_chunk(self, chunk):
        self.dset.resize(chunk.shape[0], axis=0)
        self.dset[self.row_count:] = chunk
        self.row_count += chunk.shape[0]

    def close_file(self):
        self.file.close()


def generate_sample(kappa, theta, xi, rho, dt, T):
    parameters = {
        'kappa': kappa,
        'theta': theta,
        'xi': xi,
        'rho': rho
    }
    return heston_dynamic_milstein_scheme(parameters=parameters, T=T, dt=dt)

def generate_data(parameters, number_of_samples=10, dt=0, T=0):
    pool = Pool(os.cpu_count())

    for _, kappa in enumerate(parameters['kappa']):
        for _, theta in enumerate(parameters['theta']):
            for _, rho in enumerate(parameters['rho']):
                for _, xi in enumerate(parameters['xi']):
                    engine = SimulationEnginer(kappa, theta, xi, rho, dt, T)
                    data = pool.map(engine, range(number_of_samples))
    pool.close()

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

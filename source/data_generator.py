import pickle
import glob
import os
import numpy as np
from multiprocessing import Pool

from heston import heston_dynamic_milstein_scheme
from label_mapper import LabelMapper
from file_write import FileWriter

def generate_parameter_tuples(kappas, thetas, xis, rhos):
    return tuple([(kappa, theta, xi, rho) for kappa in kappas
            for theta in thetas
                for xi in xis
                    for rho in rhos])


class DataGenerator(object):
    def __init__(self, kappas, thetas, xis, rhos, dt, T, N, chunk_size=12):
        self.dt = dt
        self.T = T
        self.time_series_length = int(T/dt)
        self.N = N
        self.lm = LabelMapper(kappas, thetas, xis, rhos)
        self.chunk_size = chunk_size
        self.param_tuples = generate_parameter_tuples(kappas, thetas, xis, rhos)

    def generate_data(self):
        pool = Pool(os.cpu_count())
        fw_training = FileWriter("training", 2, self.time_series_length)
        fw_validation = FileWriter("validation", 2, self.time_series_length)
        fw_test = FileWriter("test", 2, self.time_series_length)

        for _ in range(int(self.N/self.chunk_size)):
            data_tuple = pool.map(self.generate_sample, self.param_tuples * self.chunk_size)

            X = np.empty((np.shape(data_tuple)[0], np.shape(data_tuple[0][0])[0], np.shape(data_tuple[0][0])[1]))
            y = np.empty((np.shape(data_tuple)[0], 1), dtype=int)

            for idx, (data, label) in enumerate(data_tuple):
                X[idx] = data
                y[idx] = label

            perm = np.random.permutation(len(y))
            X = X[perm]
            y = y[perm]

            training_size = int(len(y) * 0.5)
            validation_size = int(len(y) * 0.25)

            fw_training.write_chunk(X[:training_size], y[:training_size])
            fw_validation.write_chunk(X[training_size:training_size + validation_size], y[training_size:training_size + validation_size])
            fw_test.write_chunk(X[training_size+validation_size:], y[training_size+validation_size:])

        fw_training.close_file()
        fw_validation.close_file()
        fw_test.close_file()

    def generate_sample(self, *args):
        kappa, theta, xi, rho = args[0]
        parameters = {
            'kappa': kappa,
            'theta': theta,
            'xi': xi,
            'rho': rho
        }
        r,v, _ = heston_dynamic_milstein_scheme(parameters=parameters, T=self.T, dt=self.dt)
        label = self.lm.get_label(kappa, theta, xi, rho)
        return ([r,v], label)

if __name__ == "__main__":
    # Test
    import os
    import numpy as np

    kappa = [0.2, 2, 6]
    theta = [0.1**2, 0.3**2, 0.5**2]
    rho = [-0.1, -0.5, -0.9]
    xi = [0.1, 0.3, 0.6]

    dt = 1/2
    T = 1

    dg = DataGenerator(kappa, theta, xi, rho, dt, T, 100, chunk_size=8)
    dg.generate_data()

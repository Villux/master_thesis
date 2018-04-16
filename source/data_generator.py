import pickle
import glob
import os
import numpy as np
from multiprocessing import Pool

from heston import heston_dynamic_milstein_scheme
from label_mapper import LabelMapper

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

    def generate_data(self, data_writer):
        pool = Pool(os.cpu_count())

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

            data_writer.write_chunk_training(X[:training_size], y[:training_size])
            data_writer.write_chunk_validation(X[training_size:training_size + validation_size], y[training_size:training_size + validation_size])
            data_writer.write_chunk_test(X[training_size+validation_size:], y[training_size+validation_size:])

        data_writer.close_writers()

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
    from utils.file_writer_local import FileWriterLocal
    from utils.data_writer import DataWriter

    kappa = [0.2, 2]
    theta = [0.1**2]
    xi = [0.1]
    rho = [-0.1]

    dt = 1/2
    T = 1

    param_combos = len(kappa) * len(theta) * len(rho) * len(xi)
    chunk_siz = 4
    M = 8

    dg = DataGenerator(kappa, theta, xi, rho, dt, T, M, chunk_size=chunk_siz)

    fw_train = FileWriterLocal("train")
    fw_val = FileWriterLocal("val")
    fw_test = FileWriterLocal("test")
    dw = DataWriter(fw_train, fw_val, fw_test)
    dw.close_writers = lambda : None

    dg.generate_data(dw)

    assert dg.param_tuples == ((0.2, 0.1**2, 0.1, -0.1), (2, 0.1**2, 0.1, -0.1))

    total_number_of_observations = param_combos * M
    assert dw.training.dset.shape[0] == total_number_of_observations * 0.5
    assert dw.training.dset_label.shape[0] == total_number_of_observations * 0.5
    assert dw.validation.dset.shape[0] == total_number_of_observations * 0.25
    assert dw.validation.dset_label.shape[0] == total_number_of_observations * 0.25
    assert dw.test.dset.shape[0] == total_number_of_observations * 0.25
    assert dw.test.dset_label.shape[0] == total_number_of_observations * 0.25



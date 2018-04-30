import argparse
import glob
import os

from data_generator import DataGenerator
from data_loader import load_data
from utils.data_writer_orchestrator import DataWriterOrchestrator
from utils.data_writer_h5py import DataWriterH5py
from label_mapper import LabelMapper

def generate_parameter_tuples(kappas, thetas, xis, rhos):
    return tuple([(kappa, theta, xi, rho) for kappa in kappas
            for theta in thetas
                for xi in xis
                    for rho in rhos])

def create_h5_datawriter(H, W):
    fw_training = DataWriterH5py("training", H, W)
    fw_validation = DataWriterH5py("validation", H, W)
    fw_test = DataWriterH5py("test", H, W)

    return DataWriterOrchestrator([fw_training, fw_validation, fw_test])

def run(T, dt, M):
    data_writer = create_h5_datawriter(2, int(T/dt))
    label_mapper = LabelMapper()
    # parameter_combos = ((6, 0.25, 0.6, -0.9), (2, 0.09, 0.3, -0.5), (0.2, 0.01, 0.1, -0.1),)

    # Different kappa
    # parameter_combos = ((6, 0.09, 0.3, -0.1), (2, 0.09, 0.3, -0.1), (0.2, 0.09, 0.3, -0.1),)
    # Different theta
    # parameter_combos = ((2, 0.01, 0.3, -0.5), (2, 0.09, 0.3, -0.5), (2, 0.25, 0.3, -0.5),)
    # # Different xi
    # parameter_combos = ((2, 0.09, 0.1, -0.5), (2, 0.09, 0.3, -0.5), (2, 0.09, 0.6, -0.5),)
    # # Different rho
    parameter_combos = ((2, 0.09, 0.3, -0.1), (2, 0.09, 0.3, -0.5), (2, 0.09, 0.3, -0.9),)
    for k, t, x, r in parameter_combos:
        label_mapper.add_lable(k, t, x, r)

    dg = DataGenerator(parameter_combos, label_mapper, dt, T, M)
    dg.generate_data(data_writer, [0.5, 0.25, 0.25])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-count', type=int, default=10, help="How many samples per parameter combination are created")
    parser.add_argument('--step-size', type=float, default=1/252, help="Step size (delta t)")
    parser.add_argument('--period-length', type=float, default=1, help="How many years")
    args = parser.parse_args()

    print(args)
    #try:
    files = glob.glob(f"data/*.h5")
    print(f"Remove files in data folder: {files}")

    yes = {'yes','y', 'ye', ''}
    choice = input().lower()

    if choice in yes:
        for file in files:
            os.remove(file)
    else:
        print(f"Previous data: {files} was not removed")

    run(args.period_length, args.step_size, args.sample_count)
    print('Process completed')
    # except Exception as e:
    #     print("Failed to generate data")
    #     print(f"Error: {e}")
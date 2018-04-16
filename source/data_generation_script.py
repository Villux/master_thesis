import argparse
import glob
import os

from data_generator import DataGenerator
from data_loader import load_data
from utils.data_writer_orchestrator import DataWriterOrchestrator
from utils.file_writer_h5py import FileWriterH5py

def create_h5_datawriter(H, W):
    fw_training = FileWriterH5py("training", H, W)
    fw_validation = FileWriterH5py("validation", H, W)
    fw_test = FileWriterH5py("test", H, W)

    return DataWriterOrchestrator([fw_training, fw_validation, fw_test])

def run(T, dt, M):
    kappas = [0.2, 2, 6]
    thetas = [0.1**2, 0.3**2, 0.5**2]
    xis = [0.1, 0.3, 0.6]
    rhos = [-0.1, -0.5, -0.9]

    data_writer = create_h5_datawriter(2, int(T/dt))

    dg = DataGenerator(kappas, thetas, xis, rhos, dt, T, M)
    dg.generate_data(data_writer)

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
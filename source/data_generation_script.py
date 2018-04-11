import argparse

from .data_generator import generate_data, split_data
from .data_loader import load_data

def run(T, dt, M):
    parameters = {
        'kappa': [0.2, 2, 6],
        'theta': [0.1**2, 0.3**2, 0.5**2],
        'rho': [-0.1, -0.5, -0.9],
        'xi': [0.1, 0.3, 0.6]
    }

    generate_data(parameters, number_of_samples=M, dt=dt, T=T)
    split_data(0.5, 0.25, 0.25)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-count', type=int, default=25, help="How many samples per parameter combination are created")
    parser.add_argument('--step-size', type=float, default=(1/(60*6.5))/252, help="Step size (delta t)")
    parser.add_argument('--period-length', type=float, default=2, help="How many years")
    args = parser.parse_args()

    try:
        run(args.period_length, args.step_size, args.sample_count)
        print('Process completed')
    except:
        print("Failed to generate data")
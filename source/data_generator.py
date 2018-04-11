import pickle
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

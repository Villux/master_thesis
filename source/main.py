from data_generator import generate_data
from data_loader import load_data, split_data


def main():

    parameters = {
        'kappa': [0.2, 2, 6],
        'theta': [0.1**2, 0.3**2, 0.5**2],
        'rho': [-0.1, -0.5, -0.9],
        'xi': [0.1, 0.3, 0.6]
    }

    T = 0.14
    dt = (1/(60*6.5))/252
    M = 10

    # generate_data(parameters, number_of_samples=M, dt=dt, T=T)

    # train_path, validation_path, tests_path = split_data(0.5, 0.25, 0.25)


    Xtrain, ytrain, Xval, yval, Xtest, ytest = load_data("data/training/*.p", "data/validation/*.p", "data/test/*.p")
    import ipdb; ipdb.set_trace()

    return "moi"

main()
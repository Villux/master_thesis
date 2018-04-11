import glob
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt


def get_title_from_file(filename):
    kappa, theta, xi, rho, idx = re.findall(r"[-+]?\d*\.\d+|\d+", filename)
    return f"kappa: {kappa}, theta: {theta}, xi: {xi} rho: {rho}"

def run(path, show_images):
    files = glob.glob(path)
    N = len(files)
    for im_idx in range(show_images):
        idx = np.random.randint(0, N)
        data = np.array(pickle.load(open(files[idx], "rb")))

        fig, ax1 = plt.subplots()
        returns = data[0,:]
        N = len(returns)
        price = np.ones(N)
        for i in range(1, N):
            price[i] = price[i-1] * np.exp(returns[i])
        ax1.plot(price, 'b')
        ax1.set_xlabel('time (s)')

        ax1.set_ylabel('Price development', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(data[1,:], 'r')
        ax2.set_ylabel('variance', color='r')
        ax2.tick_params('y', colors='r')

        plt.title(f"{im_idx+1}/{show_images}: " + get_title_from_file(files[idx]))
        plt.show()

if __name__ == "__main__":
    run("./data/training/*.p", 10)


import numpy as np

def heston_dynamic_milstein_scheme(parameters, T=None, dt=None, r=0, q=0, s0=1, v0=None):
    ####
    # kappa: rate at which v_t reverts to long time variance
    # theta: Long time variance
    # xi: variance of the variance
    # rho: correlation between Stock price's and volatity's Wiener process.
    # dt: step size
    # T: duration in years
    # r: risk-free rate
    # q: divident rate
    # s0: Initial price for the stock
    # v0: Initial variance
    ###

    kappa = parameters['kappa']
    theta = parameters['theta']
    xi = parameters['xi']
    rho = parameters['rho']

    n = int(T/dt) # Number of steps

    # Set long time variance as first value of variance path
    v = np.zeros(n)
    if v0:
        v[0] = v0
    else:
        v[0] = theta

    # Stock prices in logarithmic form
    s = np.zeros(n)
    s[0] = np.log(s0)

    # Random variables for Brownian motion. E.g dW_s = sqrt(dt)Z_s
    Z_s = np.random.randn(n)
    Z_v = rho * Z_s + np.sqrt(1-rho**2) * np.random.randn(n)

    for t in range(0, n-1):
        v_unsafe = v[t] + kappa * (theta - v[t]) * dt + xi * np.sqrt(v[t]) * np.sqrt(dt) * Z_v[t] + 0.25 * xi**2 * dt * (Z_v[t]**2 - 1)
        # Truncate
        v[t+1] = max(v_unsafe, 0)
        s[t+1] = s[t] + (r - q - 0.5 * v[t]) * dt + np.sqrt(v[t])* np.sqrt(dt) * Z_s[t]

    # Calculate returns
    returns = np.array([0] + [p2 - p1 for p2, p1 in zip(s, s[1:])])

    assert len(returns) == len(v) == len(s), "Lenghts of return values do not match"
    return returns, v, s


if  __name__ =='__main__':
    step_size = (1/(60*6.5))/252
    parameters = {
        'kappa': 0.2,
        'theta': 0.1**2,
        'xi': 0.1,
        'rho': -0.1
    }
    r, v, s = heston_dynamic_milstein_scheme(parameters, T=0.25, dt=step_size)

    import matplotlib.pyplot as plt
    with plt.style.context('ggplot'):
        fig, ax1 = plt.subplots()
        ax1.plot(s, 'b')
        ax1.set_xlabel('time (t)')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel('Price', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(v, 'r')
        ax2.set_ylabel('Variance', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()
        plt.show()
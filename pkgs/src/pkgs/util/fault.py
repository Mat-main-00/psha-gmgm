import numpy as np
from scipy.stats import norm


def bpt_cdf(t, mu, alpha):
    """
    Calculate the CDF of the BPT distribution.
    """
    gamma = alpha
    theta = t / mu
    term1 = norm.cdf((np.sqrt(theta) - np.sqrt(1 / theta)) / gamma)
    term2 = np.exp(2 / gamma**2) * norm.cdf(-(np.sqrt(theta) + np.sqrt(1 / theta)) / gamma)
    return term1 + term2


def sample_m(z, b, mu, ml):
    """
    Sample magnitude values based on the GR law
    """

    beta = b * np.log(10)
    m = ml - np.log(1 - z * (1 - np.exp(-beta * (mu - ml)))) / beta

    return m


if __name__ == "__main__":
    print("test")

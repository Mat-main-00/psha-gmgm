import numpy as np
from scipy.stats import norm


def bpt_cdf(t, mu, alpha):
    """
    BPT分布の累積分布関数（CDF）を計算する。
    """
    gamma = alpha
    theta = t / mu
    term1 = norm.cdf((np.sqrt(theta) - np.sqrt(1 / theta)) / gamma)
    term2 = np.exp(2 / gamma**2) * norm.cdf(-(np.sqrt(theta) + np.sqrt(1 / theta)) / gamma)
    # term1 = norm.cdf((theta - 1) / gamma)
    # term2 = np.exp(2 / gamma**2) * norm.cdf(-(theta + 1) / gamma)
    return term1 + term2


def sample_m(z, b, mu, ml):
    """
    マグニチュードをGR則に従って変換する
    :param z: 一様乱数
    :param b: GR則のB値
    :param mu: マグニチュードの上限
    :param ml: マグニチュードの下限
    :return: マグニチュード
    """

    beta = b * np.log(10)
    m = ml - np.log(1 - z * (1 - np.exp(-beta * (mu - ml)))) / beta

    return m


if __name__ == "__main__":
    print("test")

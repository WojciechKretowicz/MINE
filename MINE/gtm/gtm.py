import numpy as np
from scipy import integrate


class GTM:
    def __init__(self, d, gamma, random_state=None):
        self.d = d
        self.gamma = gamma
        self.random_state = random_state

    def generate(self, n):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        Y = np.random.binomial(1, 0.5, n)
        X = np.empty((n, self.d+1))
        for d in range(self.d):
            X[:, d] = np.random.normal(self.gamma**d * Y, 1)
        X[:, self.d] = np.random.normal(X[:, 0], 1)

        return X, Y

    @staticmethod
    def h(a):
        def f(x):
            def _exp(x):
                return np.exp(-x**2 / 2) + np.exp(-(x-a)**2 / 2)

            tmp = _exp(x) / (2 * np.sqrt(2 * np.pi))
            return tmp*np.log(tmp)

        return -integrate.quad(lambda x: f(x), min(-3*a, -10), max(3*a, 10))[0]

    def mi(self, k):
        s1 = np.sqrt((self.gamma ** (2*np.arange(0, k+1).astype(int))).sum())
        s2 = np.sqrt((self.gamma ** (2 * np.arange(0, k).astype(int))).sum())
        return GTM.h(s1) - GTM.h(s2)

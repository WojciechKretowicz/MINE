import numpy as np


class Mirror:
    def __init__(self, p=0.5):
        self.p = p

    def transform(self, x):
        return (2*np.random.binomial(1, self.p) - 1)*x


class Offset:
    def __init__(self, l=0.1, r=0.1):
        self.l = l
        self.r = r

    def transform(self, x):
        return x + np.random.uniform(self.l, self.r)


class Gamma:
    def __init__(self, l=0.5, r=2):
        self.l = l
        self.r = r

    def transform(self, x):
        return np.sign(x)*np.abs(x)**np.random.uniform(self.l, self.r)


class Tanh:
    def __init__(self, l=0.5, r=2):
        self.l = l
        self.r = r

    def transform(self, x):
        return np.tanh(np.random.uniform(self.l, self.r)*x)


class Augmentation:
    def __init__(self, transformations=None):
        if transformations is not None:
            self.transformations = np.array(transformations)
        else:
            self.transformations = np.array([Mirror(), Offset(), Gamma(), Tanh()])

    def transform(self, X, n, m=1):
        result = []
        for i in range(n):
            tmp = X.copy()
            for j in range(m):
                t = np.random.choice(len(self.transformations), X.shape[1])
                tmp = np.array([self.transformations[t[col]].transform(tmp[:, col]) for col in range(tmp.shape[1])]).T

            result.append(tmp)

        return result

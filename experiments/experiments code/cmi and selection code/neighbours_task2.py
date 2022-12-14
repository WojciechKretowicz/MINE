import numpy as np
from multiprocessing import Pool
from cmi_experiments import *
import sys
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
from MINE import GTM
from MINE import CMineOpt, MineClassif, MineOpt


if __name__ == '__main__':
    np.random.seed(77)
    print(':)')
    seeds = list(np.random.choice(1_000_000_000, size=50, replace=False))
    n_processes = 32

    with Pool(n_processes) as pool:
        r = pool.starmap(neighbours_tasks,
                     [(gamma, n, c, h, k, seed) for gamma in (0.6, 0.75, 0.9) for n in (10_000,) for c in range(1, 11) for h in (64,)
        for k in range(1, 11) for seed in seeds])

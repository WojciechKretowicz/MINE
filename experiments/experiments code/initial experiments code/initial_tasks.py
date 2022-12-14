import numpy as np
from multiprocessing import Pool
from initial_experiments import *
import sys
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
from MINE import MineClassif, MineOpt


if __name__ == '__main__':
    np.random.seed(77)
    seeds = np.random.choice(1_000_000_000, size=100, replace=False)
    n_processes = 63

    with Pool(n_processes) as pool:
        r = pool.starmap(initial_experiments,
                     [(n, seed) for n in (100, 1000, 10_000) for seed in seeds])

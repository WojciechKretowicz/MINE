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
    seeds = np.random.choice(1_000_000_000, size=50, replace=False)
    n_processes = 64

    with Pool(n_processes) as pool:
        r = pool.starmap(diff_tasks,
                     [(gamma, n, c, h, seed) for gamma in (0.6, 0.75, 0.9) for n in (1000, 5000) for c in range(1, 11) for h in (64,) for seed in seeds])

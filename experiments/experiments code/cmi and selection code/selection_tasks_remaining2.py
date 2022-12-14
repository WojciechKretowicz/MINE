import numpy as np
from multiprocessing import Pool
from selection_experiments_remaining2 import *
import sys
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
from MINE import GTM
from MINE import CMineOpt, MineClassif, MineOpt
import os


if __name__ == '__main__':
    files = os.listdir('results_selection_remaining2')
    files = set([tuple(file.split('_')[1:-1]) for file in files])

    np.random.seed(78)
    seeds = np.random.choice(1_000_000_000, size=50, replace=False)
    n_processes = 69

    with Pool(n_processes) as pool:
        r = pool.starmap(selection_tasks,
                     [(gamma, int(n), int(h), int(seed)) for gamma in (0.6, 0.75, 0.9) for n in (1000, 5000, 10_000) for h in (64,) for seed in seeds if (str(gamma), str(n), str(h), str(seed)) not in files])

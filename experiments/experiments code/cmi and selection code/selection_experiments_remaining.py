import numpy as np
import pandas as pd
from MINE import CMineDiff, GTM, MineOpt, MineClassif, CMineOpt, CMineClassif
import pickle
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model

tf.keras.backend.set_floatx('float64')


def construct_model2(input_x_shape, input_y_shape, output, hidden=64):
    inp_x = Input((input_x_shape,))
    inp_y = Input((input_y_shape,))

    x = layers.Concatenate()([inp_x, inp_y])
    x = layers.Dense(hidden, activation='tanh')(x)
    x = layers.Dense(1, activation=output)(x)

    model = Model(inputs=[inp_x, inp_y], outputs=x)
    return model


def construct_model3(input_x_shape, input_y_shape, input_z_shape, output, hidden=64):
    inp_x = Input((input_x_shape,))
    inp_y = Input((input_y_shape,))
    inp_z = Input((input_z_shape,))

    x = layers.Concatenate()([inp_x, inp_y, inp_z])
    x = layers.Dense(hidden, activation='tanh')(x)
    x = layers.Dense(1, activation=output)(x)

    model = Model(inputs=[inp_x, inp_y, inp_z], outputs=x)
    return model


def estimate_mi(x, y, method):
    def estimate_mi_opt():
        model = construct_model2(x.shape[1], y.shape[1], output='linear')

        mine = MineOpt(model)
        mine.compile(optimizer='adam')

        es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='mi', mode='max')
        history = mine.fit(x, y, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

        return mine.estimate_MI(x, y, n_shuffles=20)

    if method == 'opt':
        return estimate_mi_opt()
    else:
        raise ValueError()


def estimate_cmi(x, y, z, method, bootstrap, bootstrap_avg):
    def estimate_cmi_diff_opt():
        model1 = construct_model2(1, 1 + z.shape[1], output='linear')
        model2 = construct_model2(1, z.shape[1], output='linear')

        mine1 = MineOpt(model1)
        mine1.compile(optimizer='adam')

        mine2 = MineOpt(model2)
        mine2.compile(optimizer='adam')

        mine = CMineDiff(mine1, mine2)
        mine.compile(optimizer='adam')

        es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='mi', mode='max')
        history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

        return mine.estimate_MI(x, y, z, n_shuffles=20)

    def estimate_cmi_classif():
        model = construct_model3(x.shape[1], y.shape[1], z.shape[1], output='sigmoid')

        mine = CMineClassif(model, bootstrap=bootstrap, k=3, bootstrap_avg=bootstrap_avg)
        mine.compile(optimizer='adam', loss='binary_crossentropy')

        es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='loss', mode='min')
        history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

        return mine.estimate_MI(x, y, z, n_shuffles=20)

    def estimate_cmi_opt():
        model = construct_model3(x.shape[1], y.shape[1], z.shape[1], output='sigmoid')

        mine = CMineOpt(model, bootstrap=bootstrap, k=2, bootstrap_avg=bootstrap_avg)
        mine.compile(optimizer='adam')

        es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='mi', mode='max')
        history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

        return mine.estimate_MI(x, y, z, n_shuffles=20)

    if method == 'diff_opt':
        return estimate_cmi_diff_opt()
    elif method == 'classif':
        return estimate_cmi_classif()
    elif method == 'opt':
        return estimate_cmi_opt()
    else:
        raise ValueError()


def select(x, y, method1, method2, bootstrap, bootstrap_avg):
    y = y.reshape(-1, 1)
    selected = []
    remaining = set(list(range(x.shape[1])))
    max_ = -np.inf
    best_ = None
    for i in range(x.shape[1]):
        mi = estimate_mi(x[:, [i]], y, method1)
        if mi > max_:
            max_ = mi
            best_ = i

    selected.append(best_)
    remaining.remove(best_)

    while len(remaining) > 1:
        max_ = -np.inf
        best_ = None
        for r in remaining:
            cmi = estimate_cmi(x[:, [r]], y, x[:, selected], method2, bootstrap, bootstrap_avg)
            if cmi > max_:
                max_ = cmi
                best_ = r
        selected.append(best_)
        remaining.remove(best_)
    selected.append(remaining.pop())
    return selected


def selection_tasks(gamma, n, h, seed):
    tf.keras.utils.set_random_seed(seed)
    gtm = GTM(10, gamma)
    X, Y = gtm.generate(n)

    results = {gamma: {h: {}}}

    selected = select(X, Y, 'opt', 'diff_opt', bootstrap=None, bootstrap_avg=None)
    results[gamma][h]['diff_opt'] = selected

    selected = select(X, Y, 'opt', 'classif', bootstrap=False, bootstrap_avg=None)
    results[gamma][h]['classif'] = selected

    selected = select(X, Y, 'opt', 'opt', bootstrap=False, bootstrap_avg=None)
    results[gamma][h]['opt'] = selected

    selected = select(X, Y, 'opt', 'opt', bootstrap=True, bootstrap_avg=False)
    results[gamma][h]['opt_b'] = selected

    selected = select(X, Y, 'opt', 'opt', bootstrap=True, bootstrap_avg=True)
    results[gamma][h]['opt_b_avg'] = selected

    dt = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    with open(f'results_selection_remaining/sel_{gamma}_{n}_{h}_{seed}_{dt}', 'wb') as fd:
        pickle.dump(results, fd)


if __name__ == '__main__':
    np.random.seed(77)
    seeds = np.random.choice(10**6, 10, replace=False)
    result = {}
    k = 0
    for gamma in [0.6, 0.75, 0.9]:
        result[gamma] = {}
        for n in [1000, 5000]:
            result[gamma][n] = {}
            k += 1
            if k <= 5:
                continue
            for i in range(10):
                result[gamma][n][i] = {'diff_opt': [], 'diff_classif': [],
                                       'opt': [], 'opt_b': [], 'classif': [], 'classif_b': []}
                gtm = GTM(7, gamma, random_state=seeds[i])
                X, Y = gtm.generate(n)

                result[gamma][n][i]['diff_opt'].append(select(X, Y, 'opt', 'diff_opt', False))
                result[gamma][n][i]['diff_classif'].append(select(X, Y, 'opt', 'diff_classif', False))
                result[gamma][n][i]['opt'].append(select(X, Y, 'opt', 'opt', False))
                result[gamma][n][i]['opt_b'].append(select(X, Y, 'opt', 'opt', True))
                result[gamma][n][i]['classif'].append(select(X, Y, 'opt', 'classif', False))
                result[gamma][n][i]['classif_b'].append(select(X, Y, 'opt', 'classif', True))

            with open('checkpoints_sel/' + str(datetime.now().strftime('%Y-%m-%d_%H-%M')) + '.pkl', 'wb') as fd:
                pickle.dump(result, fd)

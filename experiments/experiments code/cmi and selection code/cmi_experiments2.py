from MINE import GTM
from MINE import CMineClassif, CMineDiff, CMineOpt, MineClassif, MineOpt
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
from datetime import datetime
import pickle


def construct_model2(input_x_shape, input_y_shape, output, hidden):
    inp_x = Input((input_x_shape,))
    inp_y = Input((input_y_shape,))

    x = layers.Concatenate()([inp_x, inp_y])
    x = layers.Dense(hidden, activation='tanh')(x)
    x = layers.Dense(1, activation=output)(x)

    model = Model(inputs=[inp_x, inp_y], outputs=x)
    return model


def construct_model3(input_x_shape, input_y_shape, input_z_shape, output, hidden):
    inp_x = Input((input_x_shape,))
    inp_y = Input((input_y_shape,))
    inp_z = Input((input_z_shape,))

    x = layers.Concatenate()([inp_x, inp_y, inp_z])
    x = layers.Dense(hidden, activation='tanh')(x)
    x = layers.Dense(1, activation=output)(x)

    model = Model(inputs=[inp_x, inp_y, inp_z], outputs=x)
    return model


def task_classif(X, Y, chosen, hidden, k):
    x = X[:, [chosen]]
    y = Y.reshape(-1, 1).astype(float)
    z = X[:, :chosen]

    model = construct_model3(1, 1, chosen, 'sigmoid', hidden)
    mine = CMineClassif(model, k=k)
    mine.compile(optimizer='adam', loss='binary_crossentropy')
    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='loss', mode='min')
    history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

    return mine, mine.estimate_MI(x, y, z, n_shuffles=20), history.history


def task_opt(X, Y, chosen, hidden, k):
    x = X[:, [chosen]]
    y = Y.reshape(-1, 1).astype(float)
    z = X[:, :chosen]

    model = construct_model3(1, 1, chosen, 'linear', hidden)
    mine = CMineOpt(model, k=k)
    mine.compile(optimizer='adam')
    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='mi', mode='max')
    history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

    return mine, mine.estimate_MI(x, y, z, n_shuffles=20), history.history


def task_diff_classif(X, Y, chosen, hidden):
    x = X[:, [chosen]]
    y = Y.reshape(-1, 1).astype(float)
    z = X[:, :chosen]

    model1 = construct_model2(1, 1+chosen, 'sigmoid', hidden)
    mine1 = MineClassif(model1)
    mine1.compile(optimizer='adam', loss='binary_crossentropy')

    model2 = construct_model2(1, chosen, 'sigmoid', hidden)
    mine2 = MineClassif(model2)
    mine2.compile(optimizer='adam', loss='binary_crossentropy')

    mine = CMineDiff(mine1, mine2)
    mine.compile(optimizer='adam')

    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='loss', mode='min')
    history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

    return mine, mine.estimate_MI(x, y, z, n_shuffles=20), history[0].history, history[1].history


def task_diff_opt(X, Y, chosen, hidden):
    x = X[:, [chosen]]
    y = Y.reshape(-1, 1).astype(float)
    z = X[:, :chosen]

    model1 = construct_model2(1, 1+chosen, 'linear', hidden)
    mine1 = MineOpt(model1)
    mine1.compile(optimizer='adam')

    model2 = construct_model2(1, chosen, 'linear', hidden)
    mine2 = MineOpt(model2)
    mine2.compile(optimizer='adam')

    mine = CMineDiff(mine1, mine2)
    mine.compile(optimizer='adam')

    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='mi', mode='max')
    history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

    return mine, mine.estimate_MI(x, y, z, n_shuffles=20), history[0].history, history[1].history


def task_classif_b(X, Y, chosen, hidden, k):
    x = X[:, [chosen]]
    y = Y.reshape(-1, 1).astype(float)
    z = X[:, :chosen]

    model = construct_model3(1, 1, chosen, 'sigmoid', hidden)
    mine = CMineClassif(model, k=k, bootstrap=True)
    mine.compile(optimizer='adam', loss='binary_crossentropy')
    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='loss', mode='min')
    history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

    return mine, mine.estimate_MI(x, y, z, n_shuffles=20), history.history


def task_opt_b(X, Y, chosen, hidden, k):
    x = X[:, [chosen]]
    y = Y.reshape(-1, 1).astype(float)
    z = X[:, :chosen]

    model = construct_model3(1, 1, chosen, 'linear', hidden)
    mine = CMineOpt(model, k=k, bootstrap=True)
    mine.compile(optimizer='adam')
    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='mi', mode='max')
    history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

    return mine, mine.estimate_MI(x, y, z, n_shuffles=20), history.history


def task_classif_b_avg(X, Y, chosen, hidden, k):
    x = X[:, [chosen]]
    y = Y.reshape(-1, 1).astype(float)
    z = X[:, :chosen]

    model = construct_model3(1, 1, chosen, 'sigmoid', hidden)
    mine = CMineClassif(model, k=k, bootstrap=True, bootstrap_avg=True)
    mine.compile(optimizer='adam', loss='binary_crossentropy')
    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='loss', mode='min')
    history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

    return mine, mine.estimate_MI(x, y, z, n_shuffles=20), history.history


def task_opt_b_avg(X, Y, chosen, hidden, k):
    x = X[:, [chosen]]
    y = Y.reshape(-1, 1).astype(float)
    z = X[:, :chosen]

    model = construct_model3(1, 1, chosen, 'linear', hidden)
    mine = CMineOpt(model, k=k, bootstrap=True, bootstrap_avg=True)
    mine.compile(optimizer='adam')
    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='mi', mode='max')
    history = mine.fit(x, y, z, batch_size=256, epochs=1000, callbacks=[es], verbose=0)

    return mine, mine.estimate_MI(x, y, z, n_shuffles=20), history.history


def neighbours_tasks(gamma, n, c, h, k, seed):
    seed = int(seed)
    tf.keras.utils.set_random_seed(seed)
    gtm = GTM(12, gamma)
    X, Y = gtm.generate(n)

    results = {gamma: {c: {h: {
        'classif': {},
        'opt': {},
        'classif_b': {},
        'opt_b': {},
        'classif_b_avg': {},
        'opt_b_avg': {}
    }}}}

    if k != 1:
        mine, mi, hist = task_classif(X, Y, c, h, k)
        results[gamma][c][h]['classif'][k] = (mi, hist)

        mine, mi, hist = task_opt(X, Y, c, h, k)
        results[gamma][c][h]['opt'][k] = (mi, hist)

    mine, mi, hist = task_classif_b(X, Y, c, h, k)
    results[gamma][c][h]['classif_b'][k] = (mi, hist)

    mine, mi, hist = task_opt_b(X, Y, c, h, k)
    results[gamma][c][h]['opt_b'][k] = (mi, hist)

    mine, mi, hist = task_classif_b_avg(X, Y, c, h, k)
    results[gamma][c][h]['classif_b_avg'][k] = (mi, hist)

    mine, mi, hist = task_opt_b_avg(X, Y, c, h, k)
    results[gamma][c][h]['opt_b_avg'][k] = (mi, hist)

    dt = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    with open(f'results2/neighbours/{gamma}_{n}_{c}_{h}_{k}_{seed}_{dt}', 'wb') as fd:
        pickle.dump(results, fd)


def diff_tasks(gamma, n, c, h, seed):
    seed = int(seed)
    tf.keras.utils.set_random_seed(seed)
    gtm = GTM(12, gamma)
    X, Y = gtm.generate(n)

    results = {gamma: {c: {h: {}}}}

    mine, mi, hist1, hist2 = task_diff_classif(X, Y, c, h)
    results[gamma][c][h]['diff_classif'] = (mi, hist1, hist2)

    mine, mi, hist1, hist2 = task_diff_opt(X, Y, c, h)
    results[gamma][c][h]['diff_opt'] = (mi, hist1, hist2)

    dt = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    with open(f'results2/diff/{gamma}_{n}_{c}_{h}_{seed}_{dt}', 'wb') as fd:
        pickle.dump(results, fd)

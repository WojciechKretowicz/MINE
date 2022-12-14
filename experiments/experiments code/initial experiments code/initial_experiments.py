import tensorflow as tf
from MINE.augmentation import *
from MINE.mi import MineOpt, MineClassif
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
from datetime import datetime
import pickle


def initial_experiments(n, seed):
    seed = int(seed)
    tf.keras.utils.set_random_seed(seed)

    datasets = prepare_datasets(n)

    result = []

    for dataset_name, dataset in datasets.items():
        if 'norm_hd' not in dataset_name:
            continue
#        result[dataset_name] = {n: {'orig':
                               #      {'DV': {'l0': {seed: None},
                              #               'l0.1': {seed: None}
                             #               },
                            #          'FD': {'l0': {seed: None},
                           #                  'l0.1': {seed: None}
                          #                  },
                         #            'classif': {seed: None}
                        #             },
                       #              'aug': {'DV': {'l0': {seed: []},
                      #                       'l0.1': {seed: []}
                     #                       },
                    #                          'FD': {'l0': {seed: []},
                   #                                  'l0.1': {seed: []}
                  #                                  },
                 #                    'classif': {seed: None}
                #                     }
               #                     }}
        print(":)")
        if 'uni' in dataset_name:
            x_ind = list(range(16))
            y_ind = list(range(16))
        else:
            if 'aug' not in dataset_name:
                size = dataset.shape[1]//2
                y_ind = list(range(size, dataset.shape[1]))
            else:
                size = dataset[0].shape[1]//2
                y_ind = list(range(size, dataset[0].shape[1]))
            x_ind = list(range(size))

            if 'norm_hd' in dataset_name:
                y_ind = y_ind[1:]

        print(dataset_name, x_ind, y_ind)

        if 'aug' not in dataset_name:
            m = fit_model_opt(dataset, 'Donsker_Varadhan', L=None, lam=0, C=0, x_ind=x_ind, y_ind=y_ind)
           # result[dataset_name][n]['orig']['DV']['l0'][seed] = m
            row = [dataset_name, 'orig', n, 'opt', 'DV', 0, seed, m[0], m[1]]
            result.append(row)

            m = fit_model_opt(dataset, 'Donsker_Varadhan', L=2, lam=0.1, C=0, x_ind=x_ind, y_ind=y_ind)
            #result[dataset_name][n]['orig']['DV']['l0.1'][seed] = m
            row = [dataset_name, 'orig', n, 'opt', 'DV', 0.1, seed, m[0], m[1]]
            result.append(row)

            m = fit_model_opt(dataset, 'f_divergence', L=None, lam=0, C=0,  x_ind=x_ind, y_ind=y_ind)
            #result[dataset_name][n]['orig']['FD']['l0'][seed] = m
            row = [dataset_name, 'orig', n, 'opt', 'FD', 0, seed, m[0], m[1]]
            result.append(row)

            m = fit_model_opt(dataset, 'f_divergence', L=2, lam=0.1, C=1, x_ind=x_ind, y_ind=y_ind)
            #result[dataset_name][n]['orig']['FD']['l0.1'][seed] = m
            row = [dataset_name, 'orig', n, 'opt', 'FD', 0.1, seed, m[0], m[1]]
            result.append(row)

            m = fit_model_classif(dataset, x_ind=x_ind, y_ind=y_ind)
            #result[dataset_name][n]['orig']['classif'][seed] = m
            row = [dataset_name, 'orig', n, 'classif', 'DV', 0, seed, m[0], m[1]]
            result.append(row)
        else:
            for i in range(10):
                m = fit_model_opt(dataset[i], 'Donsker_Varadhan', L=None, lam=0, C=0, x_ind=x_ind, y_ind=y_ind)
#                result[dataset_name][n]['aug']['DV']['l0'][seed].append(m)
                print(m)
                row = [dataset_name, 'aug', n, 'opt', 'DV', 0, seed, m[0], m[1]]
                result.append(row)

                m = fit_model_opt(dataset[i], 'Donsker_Varadhan', L=2, lam=0.1, C=0, x_ind=x_ind, y_ind=y_ind)
#                result[dataset_name][n]['aug']['DV']['l0.1'][seed].append(m)
                row = [dataset_name, 'aug', n, 'opt', 'DV', 0.1, seed, m[0], m[1]]
                result.append(row)

                m = fit_model_opt(dataset[i], 'f_divergence', L=None, lam=0, C=0, x_ind=x_ind, y_ind=y_ind)
#                result[dataset_name][n]['aug']['FD']['l0'][seed].append(m)
                row = [dataset_name, 'aug', n, 'opt', 'FD', 0, seed, m[0], m[1]]
                result.append(row)

                m = fit_model_opt(dataset[i], 'f_divergence', L=2, lam=0.1, C=1, x_ind=x_ind, y_ind=y_ind)
#                result[dataset_name][n]['aug']['FD']['l0.1'][seed].append(m)
                row = [dataset_name, 'aug', n, 'opt', 'FD', 0.1, seed, m[0], m[1]]
                result.append(row)

                m = fit_model_classif(dataset[i], x_ind=x_ind, y_ind=y_ind)
#                result[dataset_name][n]['aug']['classif'][seed] = m
                row = [dataset_name, 'aug', n, 'classif', 'DV', 0, seed, m[0], m[1]]
                result.append(row)

    dt = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    with open(f'results_retry_only_hd/{n}_{seed}_{dt}.pkl', 'wb') as fd:
        pickle.dump(result, fd)



def construct_model(input_x_shape, input_z_shape, activation):
    inp_x = Input((input_x_shape, ))
    inp_z = Input((input_z_shape, ))

    x = layers.Concatenate()([inp_x, inp_z])
    x = layers.Dense(64, activation='tanh')(x)
    x = layers.Dense(1, activation=activation)(x)
    
    model = Model(inputs=[inp_x, inp_z], outputs=x)
    return model


def create_uni(n, d):
    tmp = np.random.choice(range(d), n)
    tmp2 = np.zeros((tmp.size, tmp.max() + 1))
    tmp2[np.arange(tmp.size), tmp] = 1
    return tmp2


def fit_model_opt(dataset, approx, L, lam, C, x_ind, y_ind):
    model = construct_model(len(x_ind), len(y_ind), activation='linear')
    mine = MineOpt(model, approximation=approx, L=L, lam=lam, C=C)
    mine.compile(optimizer='adam')
    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='mi', mode='max')
    print(x_ind)
    print(y_ind)
    print(type(dataset))
    print(dataset[:3])
    history = mine.fit(dataset[:, x_ind], dataset[:, y_ind], batch_size=256, epochs=1000, callbacks=[es], verbose=0)
    return mine.estimate_MI(dataset[:, x_ind], dataset[:, y_ind], n_shuffles=20), history.history

def fit_model_classif(dataset, x_ind, y_ind):
    model = construct_model(len(x_ind), len(y_ind), activation='sigmoid')
    mine = MineClassif(model)
    mine.compile(loss='binary_crossentropy', optimizer='adam')
    es = tf.keras.callbacks.EarlyStopping(patience=20, monitor='loss', mode='min')
    history = mine.fit(dataset[:, x_ind], dataset[:, y_ind], batch_size=256, epochs=1000, callbacks=[es], verbose=0)
    return mine.estimate_MI(dataset[:, x_ind], dataset[:, y_ind], n_shuffles=20), history.history

def prepare_datasets(n):
    aug = Augmentation()

    uni = create_uni(n, 16)
    uni_aug = aug.transform(uni, n=10, m=1)

    mean = np.array([0, 1])
    cov = np.array([[1, 0], [0, 2]])
    norm_not_corr = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
    norm_not_corr_aug = aug.transform(norm_not_corr, n=10, m=1)


    mean = np.array([0, 1])
    cov = np.array([[1, 0.75], [0.75, 2]])
    norm_corr = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
    norm_corr_aug = aug.transform(norm_corr, n=10, m=1)


    cov = np.array([[ 2.97, -0.36,  1.12, -0.97,  0.07,  0.96,  2.36, -0.55,  0.88],
       [-0.36,  1.27,  0.07, -0.2 , -0.98, -0.97, -0.49,  0.46,  0.59],
       [ 1.12,  0.07,  4.21,  0.27, -2.04, -1.01,  0.45,  0.26,  0.73],
       [-0.97, -0.2 ,  0.27,  2.52, -0.57, -1.22,  0.45,  0.41, -0.89],
       [ 0.07, -0.98, -2.04, -0.57,  2.73,  2.26,  0.47, -1.12, -0.01],
       [ 0.96, -0.97, -1.01, -1.22,  2.26,  2.82,  0.78, -2.03,  0.51],
       [ 2.36, -0.49,  0.45,  0.45,  0.47,  0.78,  3.22, -0.99,  0.96],
       [-0.55,  0.46,  0.26,  0.41, -1.12, -2.03, -0.99,  2.75, -0.98],
       [ 0.88,  0.59,  0.73, -0.89, -0.01,  0.51,  0.96, -0.98,  2.2 ]])

    mean = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    #A = np.random.uniform(-1, 1, 81).reshape(9, 9)
    #cov = np.matmul(A, A.T)
    norm_hd = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
    norm_hd_aug = aug.transform(norm_hd, n=10, m=1)

    datasets = {'uni': uni, 
               'uni_aug': uni_aug, 
               'norm_not_corr': norm_not_corr, 
               'norm_not_corr_aug': norm_not_corr_aug, 
               'norm_corr': norm_corr, 
               'norm_corr_aug': norm_corr_aug, 
               'norm_hd': norm_hd,
               'norm_hd_aug': norm_hd_aug}
    return datasets



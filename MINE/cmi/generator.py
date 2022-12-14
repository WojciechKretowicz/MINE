import numpy as np
import tensorflow as tf


def knn(x, y, z, k, bootstrap=False, avg=False):
    if bootstrap:
        if not avg:
            # find k nearest neighbours' indexes based on Z
            index = tf.argsort(squared_dist(z, z))[:, 1:k+1] # skipping the closest
            # take corresponding Xs, k Xs per pair
            x_chosen = tf.reshape(tf.gather(x, index), (-1, tf.shape(x)[1]))
            # take repeating Ys and Zs for nearest Xs
            y_chosen = tf.repeat(y, k, axis=0)
            z_chosen = tf.repeat(z, k, axis=0)
            chosen = [x_chosen, y_chosen, z_chosen]
        else:
            # find k nearest neighbours' indexes based on Z
            index = tf.argsort(squared_dist(z, z))[:, 1:k+1] # skipping the closest
            # take corresponding Xs, k Xs per pair
            x_chosen = tf.math.reduce_mean(tf.gather(x, index), axis=1)
            # take repeating Ys and Zs for nearest Xs
            chosen = [x_chosen, y, z]
    else:
        n_pairs = tf.shape(x)[0] // k
        # find k nearest neighbours' indexes based on Z
        index = tf.argsort(squared_dist(z[:n_pairs], z[n_pairs:]))[:, :k]
        # take corresponding Xs, k Xs per each initial pair
        x_chosen = tf.reshape(tf.gather(x, index), (-1, tf.shape(x)[1]))
        # take repeating Ys and Zs for nearest Xs
        y_chosen = tf.repeat(y[:n_pairs], k, axis=0)
        z_chosen = tf.repeat(z[:n_pairs], k, axis=0)
        chosen = [x_chosen, y_chosen, z_chosen]

    return chosen


def squared_dist(A, B):
    A = tf.reshape(A, (tf.shape(A)[0], 1, tf.shape(A)[1]))
    B = tf.reshape(B, (1, tf.shape(B)[0], tf.shape(B)[1]))

    result = tf.norm(A - B, ord='euclidean', axis=2)

    return result

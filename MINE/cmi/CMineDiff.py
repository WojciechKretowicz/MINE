from tensorflow.keras.models import Model
from MINE.mi import Mine
import tensorflow as tf
import numpy as np


class CMineDiff(Model):
    def __init__(self, model1: Mine, model2: Mine):
        """Creates a general MINE model.

        Parameters
        ----------
        model : tf.keras.models.Model
            Statistics network
        approximation : str
            ['Donsker_Varadhan', 'f_divergence']
        K : int
            number of last iterations to use as a MI estimate on a training data
        """

        super(CMineDiff, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def call(self, inputs, training=None, mask=None):
        self.model1([inputs[0], tf.concat([inputs[1], inputs[2]], axis=1)]), self.model2([inputs[0], inputs[2]])

    def fit(self, X, Y, Z, **kwargs):
        h1 = self.model1.fit(X, tf.concat([Y, Z], axis=1), **kwargs)
        h2 = self.model2.fit(X, Z, **kwargs)
        return h1, h2

    def estimate_MI(self, x=None, y=None, z=None, n_shuffles=100):
        if x is not None and y is not None and z is not None:
            mi1 = self.model1.estimate_MI(x, np.hstack((y, z)), n_shuffles)
            mi2 = self.model2.estimate_MI(x, z, n_shuffles)

            return mi1 - mi2

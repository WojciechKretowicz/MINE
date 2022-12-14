import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np


# TODO poprawka w DV, procent calosci


class MineBase(Model):
    def __init__(self, model: Model,
                 approximation: str='Donsker_Varadhan',
                 K: int=10):
        """Creates a base for MINE model.

        Parameters
        ----------
        model : tf.keras.models.Model
            Statistics network
        approximation : str
            ['Donsker_Varadhan', 'f_divergence']
        K : int
            number of last iterations to use as a MI estimate on a training data
        """
        super(MineBase, self).__init__()
        self.model = model
        self.approximation = approximation
        self.K = K

        self._construct_approximation()

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training, mask=mask)

    def train_step(self, data):
       pass

    def _construct_approximation(self):
        if self.approximation == 'Donsker_Varadhan':
            self._second_expectation = MineBase._second_expectation_Donsker_Varadhan
            self._approximation = MineBase._Donsker_Varadhan
        elif self.approximation == 'f_divergence':
            self._second_expectation = MineBase._second_expectation_f_divergence
            self._approximation = MineBase._f_divergence
        else:
            raise ValueError("Choose as an approximation either 'Donsker_Varadhan' or 'f_divergence'.")

    @staticmethod
    def _first_expectation(T_p):
        return tf.math.reduce_mean(T_p)

    @staticmethod
    def _second_expectation_Donsker_Varadhan(T_q):
        network_output = tf.math.reduce_mean(tf.math.exp(T_q))
        return tf.math.log(network_output), network_output

    @staticmethod
    def _second_expectation_f_divergence(T_q):
        network_output = tf.math.reduce_mean(tf.math.exp(T_q - 1))
        return network_output, network_output

    @staticmethod
    def _Donsker_Varadhan(T_p, T_q):
        return MineBase._first_expectation(T_p) - MineBase._second_expectation_Donsker_Varadhan(T_q)[0]

    @staticmethod
    def _f_divergence(T_p, T_q):
        return MineBase._first_expectation(T_p) - MineBase._second_expectation_f_divergence(T_q)[0]


from MINE.MineBase import MineBase
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf


class Mine(MineBase):
    def __init__(self, model: Model,
                 approximation: str='Donsker_Varadhan',
                 K: int=10):
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
        super(Mine, self).__init__(model, approximation, K)

    def fit(self,
          x, y,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):
        return super(Mine, self).fit([x, y], None,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=verbose,
                                      callbacks=callbacks,
                                      validation_split=validation_split,
                                      validation_data=validation_data,
                                      shuffle=shuffle,
                                      class_weight=class_weight,
                                      sample_weight=sample_weight,
                                      initial_epoch=initial_epoch,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_steps=validation_steps,
                                      validation_batch_size=validation_batch_size,
                                      validation_freq=validation_freq,
                                      max_queue_size=max_queue_size,
                                      workers=workers,
                                      use_multiprocessing=use_multiprocessing
                                      )

    def estimate_MI(self, x=None, z=None, n_shuffles=100):
        if x is not None and z is not None:
            pred_joint = self([x, z], training=False)
            fes = np.repeat(self._first_expectation(pred_joint), n_shuffles)
            nos = np.empty(n_shuffles)
            for i in range(n_shuffles):
                pred_independent = self([x, tf.random.shuffle(z)], training=False)
                nos[i] = self._second_expectation(pred_independent)[1]

        elif x is None and z is None:
            fes = self.history.history['fe'][-self.K:]
            nos = self.history.history['no'][-self.K:]

        if self.approximation == 'Donsker_Varadhan':
            return np.mean(fes) - np.log(np.mean(nos))
        elif self.approximation == 'f_divergence':
            return np.mean(fes) - np.mean(nos)
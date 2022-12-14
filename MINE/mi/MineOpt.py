import tensorflow as tf
from tensorflow.keras.models import Model
from MINE.mi.Mine import Mine


class MineOpt(Mine):
    def __init__(self, model: Model,
                 approximation: str='Donsker_Varadhan',
                 K: int=10, L: float=2, lam: float=0.1, C: float=0):
        """Creates a general MINE model.

        Parameters
        ----------
        model : tf.keras.models.Model
            Statistics network
        approximation : str
            ['Donsker_Varadhan', 'f_divergence']
        K : int
            number of last iterations to use as a MI estimate on a training data
        L : float
            L in L distance used in regularization
        lam : float
            controls strength of regularization
        C : float
            second expected value will be binded to this value
        """
        super(MineOpt, self).__init__(model, approximation)
        self.L = L
        self.lam = lam
        self.C = C

        self.__construct_loss()

    def train_step(self, data):
        x, z = data[0]

        with tf.GradientTape() as tape:
            pred_joint = self([x, z], training=True)
            pred_independent = self([x, tf.random.shuffle(z)], training=True)

            # Compute our own loss
            loss, fe, se, no = self.__loss(pred_joint, pred_independent)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {'mi': fe-se, 'fe': fe, 'se': se, 'no': no}

    def __construct_loss(self):
        if self.L is None:
            self.__loss = self.__loss_without_regularization
        else:
            self.__loss = self.__loss_with_regularization

    def __loss_without_regularization(self, pred_joint, pred_independent):
        fe = self._first_expectation(pred_joint)
        se, network_output = self._second_expectation(pred_independent)

        return - fe + se, fe, se, network_output

    def __loss_with_regularization(self, pred_joint, pred_independent):
        fe = self._first_expectation(pred_joint)
        se, network_output = self._second_expectation(pred_independent)
        reg = self.lam * MineOpt.__L_distance(se, self.C, self.L)
        return - fe + se + reg, fe, se, network_output

    @staticmethod
    def __L_distance(x, y, l):
        return tf.math.pow(tf.math.reduce_sum(tf.math.pow(x - y, l)), 1/l)

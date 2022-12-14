import tensorflow as tf
from tensorflow.keras.models import Model
from MINE.cmi.CMine import CMine
from MINE.cmi.generator import knn


class CMineOpt(CMine):
    def __init__(self, model: Model,
                 approximation: str='Donsker_Varadhan',
                 k: int=5,
                 bootstrap=False,
                 bootstrap_avg=False,
                 K: int=10):
        """Creates a general MINE model.

        Parameters
        ----------
        model : tf.keras.models.Model
            Statistics network
        approximation : str
            ['Donsker_Varadhan', 'f_divergence']
        k : int
            number of nearest neighbors
        K : int
            number of last iterations to use as a MI estimate on a training data
        """
        super(CMineOpt, self).__init__(model, approximation, k, bootstrap, bootstrap_avg, K)
        self.L = None
        self.__construct_loss()

    def train_step(self, data):
        x, y, z = data[0]
        independent = knn(x, y, z, self.k, self.bootstrap)
        with tf.GradientTape() as tape:
            pred_joint = self([x, y, z], training=True)
            pred_independent = self(independent, training=True)

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
        reg = self.lam * CMineOpt.__L_distance(se, self.C, self.L)
        return - fe + se + reg, fe, se, network_output

    @staticmethod
    def __L_distance(x, y, l):
        return tf.math.pow(tf.math.reduce_sum(tf.math.pow(x - y, l)), 1/l)


if __name__ == '__main__':
    from MINE import GTM
    from MINE import CMineOpt

    import tensorflow as tf
    from tensorflow.keras import Input, layers
    from tensorflow.keras.models import Model


    def construct_model(input_x_shape, input_y_shape, input_z_shape, output):
        inp_x = Input((input_x_shape,))
        inp_y = Input((input_y_shape,))
        inp_z = Input((input_z_shape,))

        x = layers.Concatenate()([inp_x, inp_y, inp_z])
        x = layers.Dense(256, activation='tanh')(x)
        x = layers.Dense(1, activation=output)(x)

        model = Model(inputs=[inp_x, inp_y, inp_z], outputs=x)
        return model


    gtm = GTM(5, 0.8)
    X, Y = gtm.generate(1000)
    x = X[:, 3:]
    z = X[:, :3]
    Y = Y.reshape(-1, 1)

    model = construct_model(3, 1, 3, 'sigmoid')
    mine = CMineOpt(model)
    mine.compile(optimizer='adam')

    es = tf.keras.callbacks.EarlyStopping(patience=50, monitor='mi', mode='max')
    history = mine.fit([x, Y, z], None, batch_size=1024, epochs=1000, callbacks=[es])

    print(mine.estimate_MI())
    print(mine.estimate_MI(x, Y, z))

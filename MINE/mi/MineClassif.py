from tensorflow.keras.models import Model
from MINE.mi.Mine import Mine
import tensorflow as tf


class MineClassif(Mine):
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
        super(MineClassif, self).__init__(model, approximation, K)

    def classify(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training, mask=mask)

    def call(self, inputs, training=None, mask=None):
        pred = self.classify(inputs, training=training, mask=mask)
        return tf.math.log(pred / (1 - pred))

    def train_step(self, data):
        x, z = data[0]
        with tf.GradientTape() as tape:
            pred_joint = self.classify([x, z], training=True)
            pred_independent = self.classify([x, tf.random.shuffle(z)], training=True)
            pred = tf.concat([pred_joint, pred_independent], axis=0)
            y = tf.concat([tf.ones(tf.shape(pred_joint)[0]), tf.zeros(tf.shape(pred_independent)[0])], axis=0)
            # Compute our own loss
            loss = self.compiled_loss(y, pred, regularization_losses=self.losses)

            loglikelihood_ratio_joint = tf.math.log(pred_joint / (1 - pred_joint))
            loglikelihood_ratio_independent = tf.math.log(pred_independent / (1 - pred_independent))

            fe = self._first_expectation(loglikelihood_ratio_joint)
            se, no = self._second_expectation(loglikelihood_ratio_independent)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, pred)

        # Return a dict mapping metric names to current value
        result = {'mi': fe-se, 'fe': fe, 'se': se, 'no': no}
        metrics = {m.name: m.result() for m in self.metrics}
        result.update(metrics)

        return result


if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow.keras import Input, layers
    from tensorflow.keras.models import Model
    import numpy as np


    def construct_model(input_x_shape, input_z_shape, output):
        inp_x = Input((input_x_shape,))
        inp_z = Input((input_z_shape,))

        x = layers.Concatenate()([inp_x, inp_z])
        x = layers.Dense(256, activation='tanh')(x)
        x = layers.Dense(1, activation=output)(x)

        model = Model(inputs=[inp_x, inp_z], outputs=x)
        return model


    x = np.random.choice(range(16), 10000)

    model = construct_model(1, 1, 'sigmoid')
    mine = MineClassif(model, approximation='Donsker_Varadhan')
    mine.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

    es = tf.keras.callbacks.EarlyStopping(patience=50, monitor='loss', mode='min')
    history = mine.fit([x, x], None, batch_size=1024, epochs=1000, callbacks=[es])

    print(mine.estimate_MI(x, x))

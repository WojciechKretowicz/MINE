from tensorflow.keras.models import Model
from MINE.cmi.CMineDiff import CMineDiff


class CMineDiffClassif(CMineDiff):
    def __init__(self, model: Model,
                 approximation: str='Donsker_Varadhan',
                 k: int=5,
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

        super(CMineDiffClassif, self).__init__(model, approximation, k, K)

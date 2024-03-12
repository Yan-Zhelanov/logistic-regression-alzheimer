import random

import numpy as np

from utils.enums import WeightsInitType


class ParamsConfig(object):
    # Weights
    WEIGHTS_INIT_TYPE = WeightsInitType.NORMAL
    WEIGHTS_INIT_KWARGS = {'scale': 0.01, 'mean': 0}
    BIAS_ZEROS_INIT = True
    # Training params
    SEED = 0
    NUM_ITERATIONS = 1000
    LEARNING_RATE = 0.0001
    REGULARIZATION_COEFFICIENT = 0
    INPUT_VECTOR_DIMENSION = 128 * 128
    # Other
    ACTIVATION_FUNC = 'softmax'
    NUM_CLASSES = 1 if ACTIVATION_FUNC == 'sigmoid' else 2


np.random.seed(ParamsConfig.SEED)
random.seed(ParamsConfig.SEED)

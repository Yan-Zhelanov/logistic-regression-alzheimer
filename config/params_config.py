import random

import numpy as np

from utils.enums import WeightsInitType


class ParamsConfig(object):
    # Weights
    WEIGHTS_INIT_TYPE = WeightsInitType.NORMAL
    WEIGHTS_INIT_KWARGS = {'scale': 0.001, 'mean': 0}
    BIAS_ZEROS_INIT = False
    # Training params
    SEED = 0
    NUM_ITERATIONS = 10000
    ITERATIONS_WITHOUT_IMPROVEMENT = 10
    LEARNING_RATE = 0.0001
    REGULARIZATION_COEFFICIENT = 0.001
    INPUT_VECTOR_DIMENSION = 50
    # Other
    ACTIVATION_FUNC = 'softmax'
    NUM_CLASSES = 1 if ACTIVATION_FUNC == 'sigmoid' else 2


np.random.seed(ParamsConfig.SEED)
random.seed(ParamsConfig.SEED)

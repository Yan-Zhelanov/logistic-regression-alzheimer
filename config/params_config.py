import random

import numpy as np
from easydict import EasyDict

from utils.enums import WeightsInitType

params_config = EasyDict()

# Weights initialization
params_config.weights_init_type = WeightsInitType.normal
params_config.weights_init_kwargs = {'sigma': 0.01, 'mu': 0}
params_config.bias_zeros_init = True

# Training params
params_config.seed = 0
params_config.num_iterations = 1000
params_config.learning_rate = 1e-4
params_config.reg_coefficient = 0
params_config.input_vector_dimension = 128 * 128

params_config.activation_func = 'softmax'
params_config.num_classes = 1 if params_config.activation_func == 'sigmoid' else 2

# Set seed for all params initialization
np.random.seed(params_config.seed)
random.seed(params_config.seed)

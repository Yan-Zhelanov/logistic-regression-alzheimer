import os

from config.data_config import ROOT_DIR
from config.params_config import ParamsConfig


class ExperimentConfig(object):
    LOGS_DIR = os.path.join(ROOT_DIR, 'experiment_logs')
    LOSS_NAME = 'CrossEntropy'
    METRIC_NAME = 'AP'
    EXPERIMENT_NAME = f'{ParamsConfig.ACTIVATION_FUNC}_regression'
    PARAMS_DIR = os.path.join(LOGS_DIR, EXPERIMENT_NAME, 'params')
    PLOTS_DIR = os.path.join(LOGS_DIR, EXPERIMENT_NAME, 'plots')
    CHECKPOINTS_DIR = os.path.join(LOGS_DIR, EXPERIMENT_NAME, 'checkpoints')
    SAVE_MODEL_ITER = 10
    LOAD_MODEL = False
    LOAD_MODEL_EPOCH = -1
    LOAD_MODEL_PATH = os.path.join(
        CHECKPOINTS_DIR, f'checkpoint_{LOAD_MODEL_EPOCH}.pickle',
    )
    EXPERIMENT_PARAMS = {
        'learning_rate': ParamsConfig.LEARNING_RATE,
        'num_iterations': ParamsConfig.NUM_ITERATIONS,
    }


os.makedirs(ExperimentConfig.LOGS_DIR, exist_ok=True)
os.makedirs(ExperimentConfig.CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(ExperimentConfig.PARAMS_DIR, exist_ok=True)
os.makedirs(ExperimentConfig.PLOTS_DIR, exist_ok=True)

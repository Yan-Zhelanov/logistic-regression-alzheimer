import os

from easydict import EasyDict

from config.params_config import params_config

experiment_config = EasyDict()

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

experiment_config.logs_dir = os.path.join(ROOT_DIR, 'experiment_logs')
experiment_config.loss_name = 'CrossEntropy'
experiment_config.metric_name = 'AP'
experiment_config.experiment_name = f'{params_config.activation_func}_regression'

experiment_config.params_dir = os.path.join(experiment_config.logs_dir, experiment_config.experiment_name, 'params')
experiment_config.plots_dir = os.path.join(experiment_config.logs_dir, experiment_config.experiment_name, 'plots')

experiment_config.checkpoints_dir = os.path.join(experiment_config.logs_dir, experiment_config.experiment_name,
                                                 'checkpoints')
experiment_config.save_model_iter = 10

experiment_config.load_model = False
experiment_config.load_model_epoch = -1
experiment_config.load_model_path = os.path.join(experiment_config.checkpoints_dir,
                                                 f'checkpoint_{experiment_config.load_model_epoch}.pickle')

experiment_config.experiment_params = {'learning_rate': params_config.learning_rate,
                                       'num_iterations': params_config.num_iterations}

os.makedirs(experiment_config.logs_dir, exist_ok=True)
os.makedirs(experiment_config.checkpoints_dir, exist_ok=True)
os.makedirs(experiment_config.params_dir, exist_ok=True)
os.makedirs(experiment_config.plots_dir, exist_ok=True)

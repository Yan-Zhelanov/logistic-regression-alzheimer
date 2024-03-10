import os
import pickle

import plotly.graph_objects as go

from utils.enums import SetType, LoggingParamType


class ParamsLogger:
    """A class for params logging and visualization."""

    def __init__(self, config):
        self.config = config
        self.loss_history = {set_type.name: [] for set_type in SetType}
        self.metric_history = {set_type.name: [] for set_type in SetType}

        with open(os.path.join(config.logs_dir, config.experiment_name, 'experiment_params.pickle'), 'wb') as f:
            pickle.dump(config.experiment_params, f)

        if self.config.load_model:
            self.loss_history = self.get_current_params_history(LoggingParamType.loss)
            self.metric_history = self.get_current_params_history(LoggingParamType.metric)

    def log_param(self, iteration: int, set_type: SetType, param_type: LoggingParamType, metric_value: float):
        """Logs experiment parameters."""
        if param_type == LoggingParamType.loss:
            self.loss_history[set_type.name].append((iteration, metric_value))
        elif param_type == LoggingParamType.metric:
            self.metric_history[set_type.name].append((iteration, metric_value))
        else:
            raise ValueError('Unknown parameters type')

        self.save_param(set_type, param_type)

    def save_param(self, set_type: SetType, param_type: LoggingParamType):
        """Saves current state of parameters."""
        param_history = getattr(self, f'{param_type.name}_history')
        file_name = f'{set_type.name}_{param_type.name}_history.pickle'
        params_path = os.path.join(self.config.params_dir, file_name)

        with open(params_path, 'wb') as f:
            pickle.dump(param_history[set_type.name], f)

    def load_param(self, set_type: SetType, param_type: LoggingParamType):
        """Loads saved state of parameters."""
        file_name = f'{set_type.name}_{param_type.name}_history.pickle'
        params_path = os.path.join(self.config.params_dir, file_name)

        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                param_history = pickle.load(f)
            return param_history
        else:
            raise FileNotFoundError(f'{param_type.name.title()} history not found for {set_type.name} set')

    def get_current_params_history(self, param_type: LoggingParamType):
        """Gets parameters history for current experiment."""
        param_history = getattr(self, f'{param_type.name}_history')

        for set_type in (SetType.train, SetType.validation):
            if not param_history[set_type.name]:
                param_history[set_type.name] = self.load_param(set_type, param_type)
        return param_history

    def plot_params(self, param_type: LoggingParamType):
        """Visualizes parameters history."""
        param_history = self.get_current_params_history(param_type)

        fig = go.Figure()

        train_iterations, train_values = zip(*param_history[SetType.train.name])
        fig.add_trace(go.Scatter(x=train_iterations, y=train_values, mode='lines', name='Train'))

        valid_iterations, valid_values = zip(*param_history[SetType.validation.name])
        fig.add_trace(go.Scatter(x=valid_iterations, y=valid_values, mode='lines', name='Validation'))

        fig.update_layout(
            title=f'{param_type.name.title()} ({getattr(self.config, f"{param_type.name}_name")}) over iterations',
            xaxis_title='Iteration',
            yaxis_title=param_type.name.title(),
            legend_title='Set Type')

        file_name = f'{getattr(self.config, f"{param_type.name}_name").lower()}_{param_type.name}.html'
        file_path = os.path.join(self.config.plots_dir, file_name)

        fig.write_html(file_path)

        fig.show()

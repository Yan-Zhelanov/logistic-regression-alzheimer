import os
import pickle

from plotly import graph_objects as go

from config.experiment_config import ExperimentConfig
from utils.enums import LoggingParamType, SetType


class ParamsLogger(object):
    """A class for params logging and visualization."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._loss_history: dict[str, list] = {
            set_type.name: [] for set_type in SetType
        }
        self._metric_history: dict[str, list] = {
            set_type.name: [] for set_type in SetType
        }
        experiment_path = os.path.join(
            config.LOGS_DIR,
            config.EXPERIMENT_NAME,
            'experiment_params.pickle',
        )
        with open(experiment_path, 'wb') as file:
            pickle.dump(config.EXPERIMENT_PARAMS, file)
        if self._config.LOAD_MODEL:
            self._loss_history = self._get_current_params_history(
                LoggingParamType.LOSS,
            )
            self._metric_history = self._get_current_params_history(
                LoggingParamType.METRIC,
            )

    def log_param(
        self,
        iteration: int,
        set_type: SetType,
        param_type: LoggingParamType,
        metric_value: float,
    ) -> None:
        """Log experiment parameters."""
        if param_type is LoggingParamType.LOSS:
            self._loss_history[set_type.name].append((iteration, metric_value))
        elif param_type is LoggingParamType.METRIC:
            self._metric_history[set_type.name].append(
                (iteration, metric_value),
            )
        else:
            raise ValueError('Unknown parameters type')
        self._save_param(set_type, param_type)

    def plot_params(self, param_type: LoggingParamType) -> None:
        """Visualize parameters history."""
        param_history = self._get_current_params_history(param_type)
        fig = go.Figure()
        train_iterations, train_values = zip(
            *param_history[SetType.TRAIN.name],
        )
        fig.add_trace(
            go.Scatter(
                x=train_iterations, y=train_values, mode='lines', name='Train',
            ),
        )
        valid_iterations, valid_values = zip(
            *param_history[SetType.VALIDATION.name],
        )
        fig.add_trace(
            go.Scatter(
                x=valid_iterations,
                y=valid_values,
                mode='lines',
                name='Validation',
            ),
        )
        fig.update_layout(
            title=(
                f'{param_type.name.title()}'
                + f' ({getattr(self._config, f"{param_type.name}_name")})'
                + ' over iterations'
            ),
            xaxis_title='Iteration',
            yaxis_title=param_type.name.title(),
            legend_title='Set Type',
        )
        file_name = (
            f'{getattr(self._config, f"{param_type.name}_name").lower()}'
            + f'_{param_type.name}.html'
        )
        file_path = os.path.join(self._config.PLOTS_DIR, file_name)
        fig.write_html(file_path)
        fig.show()

    def _save_param(
        self, set_type: SetType, param_type: LoggingParamType,
    ) -> None:
        """Save current state of parameters."""
        if param_type is LoggingParamType.LOSS:
            param_history = self._loss_history
        else:
            param_history = self._metric_history
        file_name = (
            f'{set_type.name.lower()}_{param_type.name.lower()}_history.pickle'
        )
        params_path = os.path.join(self._config.PARAMS_DIR, file_name)
        with open(params_path, 'wb') as file:
            pickle.dump(param_history[set_type.name], file)

    def _load_param(
        self, set_type: SetType, param_type: LoggingParamType,
    ) -> dict[str, list]:
        """Load saved state of parameters."""
        file_name = (
            f'{set_type.name.lower()}_{param_type.name.lower()}_history.pickle'
        )
        params_path = os.path.join(self._config.PARAMS_DIR, file_name)
        if os.path.exists(params_path):
            with open(params_path, 'rb') as file:
                return pickle.load(file)
        raise FileNotFoundError(
            f'{param_type.name.title()} history not found for {set_type.name}'
            + ' set',
        )

    def _get_current_params_history(
        self, param_type: LoggingParamType,
    ) -> dict[str, list]:
        """Gets parameters history for current experiment."""
        param_history = getattr(self, f'_{param_type.name.lower()}_history')
        for set_type in (SetType.TRAIN, SetType.VALIDATION):
            if not param_history[set_type.name]:
                param_history[set_type.name] = self._load_param(
                    set_type, param_type,
                )
        return param_history

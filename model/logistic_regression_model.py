import os
import pickle
from typing import Union, cast

import numpy as np
from tqdm import tqdm

from config.experiment_config import ExperimentConfig
from config.params_config import ParamsConfig
from utils.enums import LoggingParamType, SetType, WeightsInitType
from utils.metrics import (
    get_accuracy_score,
    get_average_precision_score,
    get_precision_score,
    get_recall_score,
)
from utils.params_logger import ParamsLogger


class LogisticRegression(object):
    """A class for implementing and training a logistic regression model."""

    def __init__(
        self, config: ParamsConfig, experiment_config: ExperimentConfig,
    ) -> None:
        """Initialize the model.

        Args:
            config: Model and training configuration.
            experiment_config: Experiment configuration.
        """
        self._config = config
        self._num_classes = self._config.NUM_CLASSES
        self._dimension = self._config.INPUT_VECTOR_DIMENSION
        self._learning_rate = self._config.LEARNING_RATE
        self._regularization_coefficient = (
            self._config.REGULARIZATION_COEFFICIENT
        )
        self._weights = self._get_weights()
        self._bias = self._get_bias()
        # Initialization of params logger
        self._params_logger = ParamsLogger(experiment_config)
        self._start_iteration = self.prepare_model(experiment_config)
        self._checkpoints_dir = experiment_config.CHECKPOINTS_DIR
        self._save_iter = experiment_config.SAVE_MODEL_ITER
        self._best_loss: int | float = float('inf')
        self._iterations_without_improvement = 0

    def _get_weights(self) -> np.ndarray:
        """Get the initial values of the weights.

        Returns:
            numpy.ndarray of shape (K, D)
        """
        if self._config.WEIGHTS_INIT_TYPE is WeightsInitType.NORMAL:
            return self._get_weights_normal(
                **self._config.WEIGHTS_INIT_KWARGS, size=self._dimension,
            )
        return self._get_weights_uniform(
            **self._config.WEIGHTS_INIT_KWARGS, size=self._dimension,
        )

    def _get_bias(self) -> np.ndarray:
        """Get the initial values of the bias.

        Returns:
            numpy.ndarray of shape (K, 1)
        """
        if self._config.BIAS_ZEROS_INIT:
            return np.zeros((self._num_classes, 1))
        if self._config.WEIGHTS_INIT_TYPE is WeightsInitType.NORMAL:
            return self._get_weights_normal(
                **self._config.WEIGHTS_INIT_KWARGS, size=1,
            )
        return self._get_weights_uniform(
            **self._config.WEIGHTS_INIT_KWARGS, size=1,
        )

    def _get_weights_normal(
        self, mean: float = 0, scale: float = 0.01, size: int = 1,
    ) -> np.ndarray:
        """Init the weight matrix with values using a Normal distribution.

        W ~ N(mu, sigma^2),

        where:
            - mu and sigma can be defined in self.cfg.weights_init_kwargs
        """
        return np.random.normal(
            mean, scale, size=(self._num_classes, size),
        )

    def _get_weights_uniform(
        self, low: float, high: float, size: int = 1,
    ) -> np.ndarray:
        """Init the weight matrix with values using a Uniform distribution.

        W ~ U(a, b),

        where:
            - a and b can be defined in self.cfg.weights_init_kwargs
        """
        return np.random.uniform(
            low, high, size=(self._num_classes, size),
        )

    def _get_model_output(self, features: np.ndarray) -> np.ndarray:
        """Compute the model output by applying a linear transformation.

        The linear transformation is defined by the equation:
            z = W * x^T + b

            where:
                - W (a KxD matrix) represents the weight matrix,
                - x (a NxD matrix, also known as 'inputs') represents the
                    input data,
                - b (a vector of length K) represents the bias vector,
                - z represents the model output before activation.
        Args:
            features: NxD matrix.

        Returns:
            np.ndarray: KxN matrix - the model output before softmax.
        """
        return self._weights @ features.T + self._bias

    def _get_softmax_probabilities(
        self, predictions: np.ndarray,
    ) -> np.ndarray:
        """Compute the softmax function on the model output.

        The formula for softmax function is:
            y_j = e^(z_j) / Σ(i=0 to K-1) e^(z_i)

            where:
                - y_j is the softmax probability of class j,
                - z_j is the model output for class j before softmax,
                - K is the total number of classes,
                - Σ denotes summation.

        For numerical stability, subtract the max value of vector z_j before
        exponentiation: z_j = z_j - max(z_j),

        Args:
            predictions (np.ndarray): The model output before softmax.

        Returns:
            np.ndarray: KxN matrix - the softmax probabilities.
        """
        predictions = predictions - np.max(predictions, axis=0)
        exp_predictions = np.exp(predictions)
        return exp_predictions / np.sum(exp_predictions, axis=0)

    def _get_model_confidence(self, features: np.ndarray) -> np.ndarray:
        """Calculate model confidence.

        Model confidence is represented as:

            y(x, b, W) = Softmax(Wx^T + b) = Softmax(z)

            where:
                - W (a KxD matrix) represents the weight matrix,
                - x (a NxD matrix, also known as 'inputs') represents the input
                    data,
                - b (a vector of length K) represents the bias vector,
                - z represents the model output before activation,
                - y represents the model output after softmax.

        Args:
            features: NxD matrix

        Returns:
            np.ndarray:  KxN matrix - the model output after softmax.
        """
        predictions = self._get_model_output(features)
        return self._get_softmax_probabilities(predictions)

    def _get_gradient_for_weights(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        model_confidence: np.ndarray,
    ) -> np.ndarray:
        """Calculate the gradient of the cost function related to the weights.

        The gradient of the error with respect to the weights (∇w E) can be
            computed using the formula:
            ∇w E = (1 / N) * (y - t) * x,

            where:
                - y (a KxN matrix) represents the model output after softmax,
                - t (a NxK matrix) is the one-hot encoded vectors of target
                    values,
                - x (a NxD matrix, also known as 'inputs') represents the
                    input data,
                - N is the number of data points.

        For L2-regularisation:
            ∇w E = (1 / N) * (y - t) * x^T + λ * w

        Args:
            features: NxD matrix
            targets: NxK matrix
            model_confidence: KxN matrix

        Returns:
            np.ndarray: KxD matrix
        """
        return (
            (1 / features.shape[0]) * (model_confidence - targets.T)
            @ features + self._regularization_coefficient * self._weights
        )

    def _get_gradient_for_bias(
        self, targets: np.ndarray, model_confidence: np.ndarray,
    ) -> np.ndarray:
        """Calculate the gradient of the cost function related to the bias.

        The gradient of the error with respect to the bias (∇b E) can be
            computed using the formula:
            ∇b E = (1 / N) * Σ(i=0 to N-1) (y_i - t_i)

            where:
                - y (a KxN matrix) represents the model output after softmax,
                - t (a NxK matrix) is the one-hot encoded vectors of target
                    values,
                - N is the number of data points.
        Args:
            targets: NxK matrix
            model_confidence: NxK matrix

        Returns:
             np.ndarray: Kx1 matrix
        """
        return np.mean(model_confidence - targets.T, axis=1, keepdims=True)

    def _update_weights(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        model_confidence: np.ndarray,
    ) -> None:
        """Update weights and bias.

        At each iteration of gradient descent, the weights and bias are updated
        using the formula:
            w_{k+1} = w_k - γ * ∇w E(w_k)
            b_{k+1} = b_k - γ * ∇b E(b_k)

            where:
                - w_k, b_k are the current weight and bias at iteration k,
                - γ is the learning rate, determining the step size in the
                    direction of the negative gradient,
                - ∇w E(w_k) is the gradient of the cost function E with respect
                    to the weights w at iteration k,
                - ∇b E(b_k) is the gradient of the cost function E with respect
                    to the bias w at iteration k

        Args:
            features: NxD matrix
            targets: NxK matrix
            model_confidence: KxN matrix
        """
        self._weights -= self._learning_rate * self._get_gradient_for_weights(
            features, targets, model_confidence,
        )
        self._bias -= self._learning_rate * self._get_gradient_for_bias(
            targets, model_confidence,
        )

    def _train_model(
        self, features: np.ndarray, targets: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Gradient descent step.

        One step of gradient descent includes:
            1. Calculating the confidence of the model
            2. Calculating the value of the target function
            3. Updating the weights

        Returns:
            float: the value of the target function
            np.ndarray: the model output after softmax
        """
        confidence = self._get_model_confidence(features)
        loss_value = self._get_loss(
            targets, features,
        )
        self._update_weights(
            features, targets, confidence,
        )
        return loss_value, confidence

    def train(
        self,
        features_train: np.ndarray,
        targets_train: np.ndarray,
        features_valid: np.ndarray | None = None,
        targets_valid: np.ndarray | None = None,
    ) -> None:
        """Train the model using gradient descent.

        This iterative process aims to find the weights that minimize the cost
        function E(w).

        Args:
            features_train: NxD matrix
            targets_train: NxK matrix
            features_valid: MxD matrix
            targets_valid: MxK matrix
        """
        targets_train_ohe = self._convert_using_ohe(targets_train)
        targets_valid_ohe = (
            None if targets_valid is None
            else self._convert_using_ohe(targets_valid)
        )
        with_valid = features_valid is not None and targets_valid is not None
        range_bar = tqdm(range(self._config.NUM_ITERATIONS))
        for iteration in range_bar:
            loss_train, confidence = self._train_model(
                features_train, targets_train_ohe,
            )
            average_precision_train = self._get_average_precision(
                targets_train, confidence=confidence,
            )
            self._params_logger.log_param(
                iteration, SetType.TRAIN, LoggingParamType.LOSS, loss_train,
            )
            self._params_logger.log_param(
                iteration,
                SetType.TRAIN,
                LoggingParamType.METRIC,
                average_precision_train,
            )
            if with_valid:
                loss_valid, average_precision_valid = self._predict_valid(
                    features_valid=cast(np.ndarray, features_valid),
                    targets_valid=cast(np.ndarray, targets_valid),
                    targets_valid_ohe=cast(np.ndarray, targets_valid_ohe),
                )
                self._params_logger.log_param(
                    iteration,
                    SetType.VALIDATION,
                    LoggingParamType.LOSS,
                    loss_valid,
                )
                self._params_logger.log_param(
                    iteration,
                    SetType.VALIDATION,
                    LoggingParamType.METRIC,
                    average_precision_valid,
                )
                range_bar.set_description(
                    f'Train Loss: {loss_train};'
                    + f'\tTrain AP:{average_precision_train}; '
                    + f'\tValid AP:{average_precision_valid}\t',
                )
                self._update_iterations_without_improvement(
                    iteration, loss_train,
                )
            else:
                range_bar.set_description(
                    f'Train loss: {loss_train};'
                    + f'\tTrain AP:{average_precision_train}; ',
                )
            print()
            if self._is_stop_needed():
                break

    def _predict_valid(
        self,
        features_valid: np.ndarray,
        targets_valid: np.ndarray,
        targets_valid_ohe: np.ndarray,
    ) -> tuple[float, float]:
        return self._get_loss(
            targets_valid_ohe, features_valid,
        ), self._get_average_precision(
            targets_valid, features_valid,
        )

    def _get_average_precision(
        self,
        targets: np.ndarray,
        features: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
    ) -> float:
        if features is None and confidence is None:
            raise ValueError(
                'Features and confidence cannot be None at the same time!',
            )
        if confidence is None:
            confidence = self._get_model_confidence(cast(np.ndarray, features))
        return get_average_precision_score(targets, confidence[-1, :])

    def _update_iterations_without_improvement(
        self,
        iteration: int,
        loss_valid: float,
    ) -> None:
        if loss_valid < self._best_loss:
            self._iterations_without_improvement = 0
            self._best_loss = loss_valid
            self._save(f'best_loss_{iteration}.pickle')
        else:
            self._iterations_without_improvement += 1

    def _is_stop_needed(self) -> bool:
        if self._config.ITERATIONS_WITHOUT_IMPROVEMENT > 0:
            return (
                self._iterations_without_improvement
                >= self._config.ITERATIONS_WITHOUT_IMPROVEMENT
            )
        return False

    def _get_loss(
        self,
        targets: np.ndarray,
        features: Union[np.ndarray, None] = None,
        prediction_without_softmax: Union[np.ndarray, None] = None,
        model_confidence: Union[np.ndarray, None] = None,
    ) -> float:
        """Target function.

        Cross-Entropy Loss:
            E = - (1 / N) Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * ln(y_k (x_i)),

            where:
                - N is the size of the data set,
                - K is the number of classes,
                - t_{ik} is the value from OHE target matrix for data point i
                    and class k,
                - y_k (x_i) is model output after softmax for data point i and
                    class k.

        Numerically stable formula:
            E = (1 / N)
                Σ(i=0 to N-1) Σ(k=0 to K-1)
                t_ik * (ln(Σ(l=0 to K-1) e^((z_il - c_i)) - (z_ik - c_i)),

            where:
                - N is the size of the data set,
                - K is the number of classes,
                - t_{ik} is the value from OHE target matrix for data point i
                    and class k,
                - z_{il} is the model output before softmax for data point i
                    and class l,
                - z is the model output before softmax (matrix z),
                - c_i is maximum value for each data point i in vector z_i.

        Parameters:
            targets (np.ndarray): The target data.
            inputs (Union[np.ndarray, None]: The input data.
            predictions_without_softmax (Union[np.ndarray, None]): The model
                output before softmax. If None, it will be computed.
            model_confidence (Union[np.ndarray, None]): The model output after
                softmax. If None, it will be computed.

        Returns:
            float: The value of the target function.
        """
        if model_confidence is None and features is None:
            raise ValueError(
                'Model confidence and features cannot be None at the same time'
                + ' when calculating the loss.',
            )
        if model_confidence is None:
            model_confidence = self._get_model_confidence(
                cast(np.ndarray, features),
            )
        if prediction_without_softmax is None:
            prediction_without_softmax = self._get_model_output(
                cast(np.ndarray, features),
            )
        max_prediction = np.max(prediction_without_softmax, axis=0)
        prediction_without_softmax -= max_prediction
        log_predictions = np.log(
            np.sum(np.exp(prediction_without_softmax), axis=0),
        )
        errors = targets.T * (
            log_predictions - prediction_without_softmax
        )
        return np.mean(np.sum(errors, axis=0))

    def _get_metrics(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        model_confidence: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Metrics calculation."""
        if model_confidence is None:
            model_confidence = self._get_model_confidence(features)
        predictions = np.argmax(model_confidence, axis=0)
        return {
            'accuracy': get_accuracy_score(targets, predictions),
            'average_precision': get_average_precision_score(
                targets, model_confidence,
            ),
            'precision': get_precision_score(targets, predictions),
            'recall': get_recall_score(targets, predictions),
        }

    def _convert_using_ohe(self, targets: np.ndarray) -> np.ndarray:
        """Create matrix of one-hot encoding vectors for input targets.

        One-hot encoding vector representation:
            t_i^(k) = 1 if k = t_i otherwise  0,

            where:
                - k in [0, self.k-1],
                - t_i - target class of i-sample.

        Args:
            targets: NxK matrix

        Returns:
            np.ndarray: NxK matrix
        """
        targets = targets.astype(int)
        one_hot_encoded_targets = np.zeros(
            (targets.shape[0], self._num_classes),
        )
        one_hot_encoded_targets[np.arange(targets.shape[0]), targets] = 1
        return one_hot_encoded_targets

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Return model prediction."""
        model_confidence = self._get_model_confidence(features)
        return np.argmax(model_confidence, axis=0)

    def _save(self, filepath: str) -> None:
        """Save trained model."""
        with open(os.path.join(self._checkpoints_dir, filepath), 'wb') as file:
            pickle.dump((self._weights, self._bias), file)

    def load(self, filepath: str) -> None:
        """Load trained model."""
        with open(filepath, 'rb') as file:
            self._weights, self._bias = pickle.load(file)

    def prepare_model(self, experiment_config: ExperimentConfig) -> int:
        """Prepare model.

        Checkpoint loading (if needed) and start iteration set up.

        Args:
            experiment_config: Experiment configuration.

        Returns:
            int: The start iteration.
        """
        start_iteration = 0
        if experiment_config.LOAD_MODEL:
            try:
                self.load(experiment_config.LOAD_MODEL_PATH)
                print(
                    'Model loaded successfully from'
                    + f' {experiment_config.LOAD_MODEL_PATH}',
                )
                start_iteration = experiment_config.LOAD_MODEL_EPOCH + 1
            except FileNotFoundError:
                print(
                    'Model file not found at'
                    + f' {experiment_config.LOAD_MODEL_PATH}.'
                    + ' Using init weights.',
                )
            except Exception as error:
                print(
                    f'An error occurred while loading the model: {str(error)}.'
                    + ' Using init weight.',
                )
        return start_iteration

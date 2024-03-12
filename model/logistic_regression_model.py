import os
import pickle
from typing import Union

import numpy as np

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
        # Initialization of weights and bias
        self._weights = None  # numpy.ndarray of shape (K, D)
        # Initialize bias with zeros if self.cfg.bias_zeros_init is True
        # numpy.ndarray of shape (K, 1)
        self._bias = np.zeros((self._num_classes, 1))
        if self._config.WEIGHTS_INIT_TYPE is WeightsInitType.NORMAL:
            self._init_weights_normal(**self._config.WEIGHTS_INIT_KWARGS)
        else:
            self._init_weights_uniform(**self._config.WEIGHTS_INIT_KWARGS)
        # Initialization of params logger
        self._params_logger = ParamsLogger(experiment_config)
        self._start_iteration = self.prepare_model(experiment_config)
        self._checkpoints_dir = experiment_config.CHECKPOINTS_DIR
        self._save_iter = experiment_config.SAVE_MODEL_ITER

    def _init_weights_normal(
        self, mean: float = 0, scale: float = 0.01,
    ) -> np.ndarray:
        """Init the weight matrix with values using a Normal distribution.

        W ~ N(mu, sigma^2),

        where:
            - mu and sigma can be defined in self.cfg.weights_init_kwargs
        """
        return np.random.normal(
            mean, scale, size=(self._num_classes, self._dimension),
        )

    def _init_weights_uniform(self, low: float, high: float) -> np.ndarray:
        """Init the weight matrix with values using a Uniform distribution.

        W ~ U(a, b),

        where:
            - a and b can be defined in self.cfg.weights_init_kwargs
        """
        return np.random.uniform(
            low, high, size=(self._num_classes, self._dimension),
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

    def __softmax(self, model_output: np.ndarray) -> np.ndarray:
        """Computes the softmax function on the model output.

        The formula for softmax function is:
            y_j = e^(z_j) / Σ(i=0 to K-1) e^(z_i)

            where:
                - y_j is the softmax probability of class j,
                - z_j is the model output for class j before softmax,
                - K is the total number of classes,
                - Σ denotes summation.

        For numerical stability, subtract the max value of vector z_j before exponentiation:
            z_j = z_j - max(z_j),

        Args:
            model_output (np.ndarray): The model output before softmax

        Returns:
            np.ndarray: KxN matrix - the softmax probabilities
        """
        # TODO: Implement numerically stable softmax
        raise NotImplementedError

    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        """Calculates model confidence.

        Model confidence is represented as:

            y(x, b, W) = Softmax(Wx^T + b) = Softmax(z)

            where:
                - W (a KxD matrix) represents the weight matrix,
                - x (a NxD matrix, also known as 'inputs') represents the input data,
                - b (a vector of length K) represents the bias vector,
                - z represents the model output before activation,
                - y represents the model output after softmax.

        Args:
            inputs: NxD matrix

        Returns:
            np.ndarray:  KxN matrix - the model output after softmax.
        """
        z = self._get_model_output(inputs)
        y = self.__softmax(z)
        return y

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the cost function with respect to the weights.

        The gradient of the error with respect to the weights (∇w E) can be computed using the formula:
            ∇w E = (1 / N) * (y - t) * x,

            where:
                - y (a KxN matrix) represents the model output after softmax,
                - t (a NxK matrix) is the one-hot encoded vectors of target values,
                - x (a NxD matrix, also known as 'inputs') represents the input data,
                - N is the number of data points.

        For L2-regularisation:
            ∇w E = (1 / N) * (y - t) * x^T + λ * w

        Args:
            inputs: NxD matrix
            targets: NxK matrix
            model_confidence: KxN matrix

        Returns:
             np.ndarray: KxD matrix
        """
        # TODO: Implement this method using matrix operations in numpy. Do not use loops
        # TODO: Add regularization
        raise NotImplementedError

    def __get_gradient_b(self, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the cost function with respect to the bias.

        The gradient of the error with respect to the bias (∇b E) can be computed using the formula:

            ∇b E = (1 / N) * Σ(i=0 to N-1) (y_i - t_i)

            where:
                - y (a KxN matrix) represents the model output after softmax,
                - t (a NxK matrix) is the one-hot encoded vectors of target values,
                - N is the number of data points.
        Args:
            targets: NxK matrix
            model_confidence: NxK matrix

        Returns:
             np.ndarray: Kx1 matrix
        """
        # TODO: Implement this method using matrix operations in numpy. Do not use loops
        raise NotImplementedError

    def __weights_update(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        """Updates weights and bias.

        At each iteration of gradient descent, the weights and bias are updated using the formula:

            w_{k+1} = w_k - γ * ∇w E(w_k)
            b_{k+1} = b_k - γ * ∇b E(b_k)

            where:
                - w_k, b_k are the current weight and bias at iteration k,
                - γ is the learning rate, determining the step size in the direction of the negative gradient,
                - ∇w E(w_k) is the gradient of the cost function E with respect to the weights w at iteration k,
                - ∇b E(b_k) is the gradient of the cost function E with respect to the bias w at iteration k,
        """
        # TODO: Implement this method using self.__get_gradient_b and self.__get_gradient_w
        raise NotImplementedError

    def __gradient_descent_step(self, inputs: np.ndarray, targets: np.ndarray):
        """Gradient descent step.

        One step of gradient descent includes:
            1. Calculating the confidence of the model
            2. Calculating the value of the target function
            3. Updating the weights

        Returns:
            float: the value of the target function
            np.ndarray: the model output after softmax
        """
        # TODO: Implement this method using self.get_model_confidence,
        #       self.__target_function_value and self.__weights_update
        raise NotImplementedError

    def train(
        self, inputs_train: np.ndarray, targets_train: np.ndarray,
        inputs_valid: Union[np.ndarray, None] = None, targets_valid: Union[np.ndarray, None] = None,
    ):
        """Training with gradient descent.

        This iterative process aims to find the weights that minimize the cost function E(w).
        """
        # TODO:
        #  For each iteration from self.start_iteration to self.cfg.num_iterations:
        #       - make gradient descent step (using self.__gradient_descent_step)
        #       - compute metrics using the self.compute_metrics function
        #       - logging value of cost function and metrics with self.params_logger.log_param()
        #  Before training convert targets to one-hot encoded vectors using the self.one_hot_encoding function
        #  Also you can compute metrics in validation set using inputs_valid and targets_valid args and
        #  save best model params with self.save
        #  You can add early stopping using metric (average precision or other) on the validation set:
        #       - if there is no improvement after a given number of iterations, training can be stopped
        raise NotImplementedError

    def __target_function_value(
        self,
        targets: np.ndarray,
        inputs: Union[np.ndarray, None] = None,
        z: Union[np.ndarray, None] = None,
        model_confidence: Union[np.ndarray, None] = None,
    ) -> float:
        """Target function.

        Cross-Entropy Loss:
            E = - (1 / N) Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * ln(y_k (x_i)),

            where:
                - N is the size of the data set,
                - K is the number of classes,
                - t_{ik} is the value from OHE target matrix for data point i and class k,
                - y_k (x_i) is model output after softmax for data point i and class k.

        Numerically stable formula:
            E = (1 / N) Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * (ln(Σ(l=0 to K-1) e^((z_il - c_i)) - (z_ik - c_i)),

            where:
                - N is the size of the data set,
                - K is the number of classes,
                - t_{ik} is the value from OHE target matrix for data point i and class k,
                - z_{il} is the model output before softmax for data point i and class l,
                - z is the model output before softmax (matrix z),
                - c_i is maximum value for each data point i in vector z_i.

        Parameters:
            targets (np.ndarray): The target data.
            inputs (Union[np.ndarray, None]: The input data.
            z (Union[np.ndarray, None]): The model output before softmax. If None, it will be computed.
            model_confidence (Union[np.ndarray, None]): The model output after softmax. If None, it will be computed.

        Returns:
            float: The value of the target function.
        """
        # TODO: Implement this function, it is possible to do it without loop using numpy
        raise NotImplementedError

    def compute_metrics(
        self, inputs: np.ndarray, targets: np.ndarray,
        model_confidence: Union[np.ndarray, None] = None,
    ):
        """Metrics calculation."""
        # TODO: Add calculation of metrics, e.g., accuracy, precision, recall, average precision, confusion matrix
        raise NotImplementedError

    def one_hot_encoding(self, targets):
        """Creates matrix of one-hot encoding vectors for input targets.

        One-hot encoding vector representation:
            t_i^(k) = 1 if k = t_i otherwise  0,

            where:
                - k in [0, self.k-1],
                - t_i - target class of i-sample.
        """
        # TODO: Implement this function, it is possible to do it without loop using numpy
        raise NotImplementedError

    def __call__(self, inputs: np.ndarray):
        """Returns model prediction."""
        model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        return predictions

    def save(self, filepath):
        """Saves trained model."""
        with open(os.path.join(self._checkpoints_dir, filepath), 'wb') as f:
            pickle.dump((self._weights, self._bias), f)

    def load(self, filepath):
        """Loads trained model."""
        with open(filepath, 'rb') as f:
            self._weights, self._bias = pickle.load(f)

    def prepare_model(self, experiment_config):
        """Prepares model: checkpoint loading (if needed) and start iteration set up."""
        start_iteration = 0
        if experiment_config.load_model:
            try:
                self.load(experiment_config.load_model_path)
                print(
                    f'Model loaded successfully from {experiment_config.load_model_path}',
                )
                start_iteration = experiment_config.load_model_epoch + 1
            except FileNotFoundError:
                print(
                    f'Model file not found at {experiment_config.load_model_path}. Using init weights.',
                )
            except Exception as e:
                print(
                    f'An error occurred while loading the model: {str(e)}. Using init weight.',
                )
        return start_iteration

import numpy as np

from config.data_config import DataConfig
from utils.enums import PreprocessingType

IntOrFloat = int | float


class ImageDataPreprocessing(object):
    """A class for data preprocessing."""

    def __init__(
        self,
        preprocess_type: PreprocessingType,
        data_config: DataConfig,
    ) -> None:
        self._preprocess_type = preprocess_type
        self._data_config = data_config
        self._min: IntOrFloat = 0
        self._max: IntOrFloat = 0
        self._mean: IntOrFloat = 0
        self._std: IntOrFloat = 0

    def fit(self, features: np.ndarray) -> None:
        """Initialize preprocessing function on training data.

        Args:
            features: feature array.
        """
        flattened_features = self._flatten(features)
        self._mean = np.mean(flattened_features)
        self._std = np.std(flattened_features)
        self._min = np.min(flattened_features)
        self._max = np.max(flattened_features)

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """Return preprocessed data."""
        flattened_features = self._flatten(features)
        if self._preprocess_type is PreprocessingType.NORMALIZATION:
            return self._normalization(flattened_features)
        return self._standardization(flattened_features)

    def _normalization(self, features: np.ndarray) -> np.ndarray:
        """Transform features by scaling each pixel.

        Scaling occurs in a range [a, b] with the min and max params.

        Args:
            features: Array of images.

        Returns:
            np.ndarray: Normilized features.
        """
        lower_bound = self._data_config.PREPROCESS_LOWER_BOUND
        upper_bound = self._data_config.PREPROCESS_UPPER_BOUND
        # TODO: implement data normalization
        #       normalized_x = a + (b - a) * (x - self.params['min']) / (self.params['max'] - self.params['min']),
        #       where a = self.params['a'], b = self.params['b']
        return features

    def _standardization(self, features: np.ndarray) -> np.ndarray:
        """Standardize features with the mean and std params.

        Args:
            features: Array of images.

        Returns:
            np.ndarray: Standardized features.
        """
        # TODO: implement data standardization
        #       standardized_x = (x - self.params['mean']) / self.params['std']
        return features

    def _flatten(self, features: np.ndarray) -> np.ndarray:
        """Reshape features into a matrix of shape (N, HxW).

        Args:
            features: Array of images.

        Returns:
            np.ndarray: Flattened features.
        """
        # TODO: Reshape x from (N, H, W) to shape (N, HxW)
        return features

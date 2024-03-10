import numpy as np

from utils.enums import PreprocessingType


class ImageDataPreprocessing:
    """A class for data preprocessing."""

    def __init__(self, preprocess_type: PreprocessingType, preprocess_params=None):
        self.preprocess_type = preprocess_type

        # A dictionary with the following keys and values:
        #    - {'min': min values, 'max': max values} when preprocess_type is PreprocessingType.normalization
        #    - {'mean': mean values, 'std': std values} when preprocess_type is PreprocessingType.standardization
        self.params = {}

        # Initialization of additional parameters
        if isinstance(preprocess_params, dict):
            self.params.update({key: value for key, value in preprocess_params.items()})

        # Select the preprocess function according to self.preprocess_type
        self.preprocess_func = getattr(self, self.preprocess_type.name)

    def normalization(self, x: np.ndarray, init=False):
        """Transforms x by scaling each pixel to a range [a, b] with self.params['min'] and self.params['max']

        Args:
            x: array of images
            init: initialization flag

        Returns:
            normalized_x (numpy.array)
        """
        if init:
            # TODO: calculate min and max for all pixels
            #       store the values in self.params['min'] and self.params['max']
            pass

        a = self.params.get('a', -1)
        b = self.params.get('b', 1)

        # TODO: implement data normalization
        #       normalized_x = a + (b - a) * (x - self.params['min']) / (self.params['max'] - self.params['min']),
        #       where a = self.params['a'], b = self.params['b']
        raise NotImplementedError

    def standardization(self, x: np.ndarray, init=False):
        """Standardizes x with self.params['mean'] and self.params['std']

        Args:
            x: array of images
            init: initialization flag

        Returns:
            standardized_x (numpy.array)
        """
        if init:
            # TODO: calculate mean and std of all pixels in x with np.mean, np.std
            #       store the values in self.params['mean'] and self.params['std']
            pass

        # TODO: implement data standardization
        #       standardized_x = (x - self.params['mean']) / self.params['std']
        raise NotImplementedError

    def flatten(self, x: np.ndarray):
        """Reshaping x into a matrix of shape (N, HxW)"""
        # TODO: Reshape x from (N, H, W) to shape (N, HxW)
        raise NotImplementedError

    def train(self, x: np.ndarray):
        """Initializes preprocessing function on training data."""
        flattened_x = self.flatten(x)
        return self.preprocess_func(flattened_x, init=True)

    def __call__(self, x: np.ndarray):
        """Returns preprocessed data."""
        if self.params is None:
            raise Exception(f"{self.preprocess_type.name} instance is not trained yet. Please call 'train' first.")
        flattened_x = self.flatten(x)
        return self.preprocess_func(flattened_x, init=False)

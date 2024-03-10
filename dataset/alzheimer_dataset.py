import os

import cv2
from PIL import Image
from skimage import io
import numpy as np

from utils.common_functions import read_dataframe_file
from utils.enums import SetType
from utils.preprocessing import ImageDataPreprocessing


class AlzheimerDataset:
    """A class for the Alzheimer dataset. This class reads the data and preprocesses it."""

    def __init__(self, config):
        """Initializes the Alzheimer dataset class instance."""
        self.config = config

        # Reading an annotation file that contains the image path, set_type, and target values for the entire dataset
        self.annotation = read_dataframe_file(os.path.join(config.path_to_data, config.annot_filename))

        # Preprocessing class initialization
        self.preprocessing = ImageDataPreprocessing(config.preprocess_type, config.preprocess_params)

        # Reads the data
        self.data = {}
        for set_type in SetType:
            self.data[set_type.name] = self.data_preprocessing(set_type)

    def data_preprocessing(self, set_type):
        """Preparing data.

        Args:
            set_type: data set_type from SetType

        Returns:
            A dict with the following data:
                {'features: images (numpy.ndarray), 'targets': targets (numpy.ndarray), 'paths': list of paths}
        """
        # TODO:
        #  1) Get the rows from the self.annotation-dataframe with 'set_type' == set_type.name
        #  2) Drop duplicates from dataframe if necessary (except when set_type is SetType.test)
        #  3) For each row of the dataframe:
        #       - read the image by 'path' column
        #       - convert it to GRAYSCALE mode
        #       - transform it to numpy.ndarray with dtype=np.float64
        #       - add the read image to the list of images
        #       You can use, for example, Pillow, opencv or scikit-image libraries to work with images
        #  4) Stack images-list to numpy.ndarray with shape (N, H, W), where
        #       - N - number of samples
        #       - H, W - images height and width
        #  5) Apply self.preprocessing to images according SetType:
        #          if set_type is SetType.train, call self.preprocessing.train(),
        #          otherwise call self.preprocessing()
        #  6) Create targets from columns 'target' (except when set_type is SetType.test):
        #       - transform to numpy.ndarray with dtype=np.int64
        #  7) Return arrays of images, targets and paths as a dict
        images = []
        raise NotImplementedError

    def __call__(self, set_type: str):
        """Returns preprocessed data."""
        return self.data[set_type]

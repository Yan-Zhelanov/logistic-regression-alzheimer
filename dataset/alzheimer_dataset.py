import os

import cv2
import numpy as np
from PIL import Image
from skimage import io

from config.data_config import DataConfig
from utils.common_functions import read_dataframe_file
from utils.enums import SetType
from utils.preprocessing import ImageDataPreprocessing


class AlzheimerDatasetPreprocessor(object):
    """A class for the Alzheimer dataset.

    This class reads the data and preprocesses it.
    """

    def __init__(self, data_config: DataConfig) -> None:
        """Initializes the Alzheimer dataset class instance."""
        self._data_config = data_config
        self._annotation = read_dataframe_file(
            os.path.join(
                data_config.PATH_TO_DATA, data_config.ANNOTATION_FILENAME,
            ),
        )
        self._preprocessing = ImageDataPreprocessing(
            data_config.PREPROCESS_TYPE, data_config.PREPROCESS_PARAMS,
        )
        self._data = {}
        for set_type in (SetType.TRAIN, SetType.VALIDATION, SetType.TEST):
            self._data[set_type] = self._data_preprocessing(set_type)

    def get_preprocessed_data(self, set_type: SetType):
        """Return preprocessed data."""
        return self._data[set_type]

    def _data_preprocessing(self, set_type: SetType) -> dict[str, np.ndarray]:
        """Preparing data.

        Args:
            set_type: data set_type from SetType

        Returns:
            dict[str, np.ndarray]: A dict with the following data - {
                'features: images (numpy.ndarray),
                'targets': targets (numpy.ndarray),
                'paths': list of paths,
            }
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
        # images = []
        return {}

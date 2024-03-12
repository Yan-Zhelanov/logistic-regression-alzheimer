import os

import cv2
import numpy as np

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
            data_config.PREPROCESS_TYPE, data_config,
        )
        self._data = {}
        for set_type in (SetType.TRAIN, SetType.VALIDATION, SetType.TEST):
            self._data[set_type] = self._data_preprocessing(set_type)

    def get_preprocessed_data(self, set_type: SetType):
        """Return preprocessed data."""
        return self._data[set_type]

    def _data_preprocessing(
        self, set_type: SetType,
    ) -> dict[str, np.ndarray | list[str]]:
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
        annotation = self._annotation[
            self._annotation['set'] == set_type.name.lower()
        ]
        if set_type is not SetType.TEST:
            annotation = annotation.drop_duplicates()
        images = []
        for _, image_info in annotation.iterrows():
            images.append(
                cv2.imread(
                    f'{self._data_config.PATH_TO_DATA}/{image_info["path"]}',
                    cv2.IMREAD_GRAYSCALE,
                ),
            )
        converted_images = np.array(images, dtype=np.float64)
        if set_type is SetType.TRAIN:
            self._preprocessing.fit(converted_images)
        converted_images = self._preprocessing.preprocess(converted_images)
        return {
            'features': converted_images,
            'paths': annotation['path'].to_list(),
            **(
                {'targets': np.array(annotation['target'])}
                if set_type is not SetType.TEST else {}
            ),
        }

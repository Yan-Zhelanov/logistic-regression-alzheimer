import os

from utils.enums import PreprocessingType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class DataConfig(object):
    PATH_TO_DATA = os.path.join(ROOT_DIR, 'dataset', 'images')
    ANNOTATION_FILENAME = 'data_info.csv'
    PREPROCESS_TYPE = PreprocessingType.NORMALIZATION
    PREPROCESS_LOWER_BOUND = -1
    PREPROCESS_UPPER_BOUND = 1

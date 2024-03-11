import os

from utils.enums import PreprocessingType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class Config(object):
    PATH_TO_DATA = os.path.join(ROOT_DIR, 'dataset')
    ANNOTATION_FILENAME = 'data_info.csv'
    PREPROCESS_TYPE = PreprocessingType.NORMALIZATION
    PREPROCESS_PARAMS = {'a': -1, 'b': 1}

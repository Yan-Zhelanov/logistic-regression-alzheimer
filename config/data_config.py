import os

from easydict import EasyDict

from utils.enums import PreprocessingType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

data_config = EasyDict()

# Path to the directory with dataset files
data_config.path_to_data = os.path.join(ROOT_DIR, 'data', 'alzheimer')
data_config.annot_filename = 'data_info.csv'

data_config.preprocess_type = PreprocessingType.normalization  # Normalization, standardization
data_config.preprocess_params = {'a': -1, 'b': 1}

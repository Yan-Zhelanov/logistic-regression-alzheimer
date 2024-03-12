import pandas as pd

from config.data_config import DataConfig
from config.experiment_config import ExperimentConfig
from config.params_config import ParamsConfig
from dataset.alzheimer_dataset import AlzheimerDatasetPreprocessor
from model.logistic_regression_model import LogisticRegression
from utils.enums import SetType


def train() -> None:
    dataset = AlzheimerDatasetPreprocessor(DataConfig())
    train_data = dataset.get_preprocessed_data(SetType.TRAIN)
    valid_data = dataset.get_preprocessed_data(SetType.VALIDATION)
    model = LogisticRegression(ParamsConfig(), ExperimentConfig())
    model.train(
        train_data['features'],
        train_data['targets'],
        valid_data['features'],
        valid_data['targets'],
    )


def predict() -> None:
    # TODO: Set experiment_config.load_model and
    # experiment_config.load_model_epoch before evaluating on the test dataset
    dataset = AlzheimerDatasetPreprocessor(DataConfig())
    test_data = dataset.get_preprocessed_data(SetType.TEST)
    model = LogisticRegression(ParamsConfig(), ExperimentConfig())
    test_predictions = model._get_model_confidence(test_data['features'])[1, :]
    test_results_df = pd.DataFrame(
        {'ID': test_data['paths'], 'prediction': test_predictions},
    )
    test_results_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    train()

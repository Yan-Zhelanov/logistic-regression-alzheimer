import numpy as np
import pandas as pd

from config.data_config import data_config
from config.experiment_config import experiment_config
from config.params_config import params_config
from dataset.alzheimer_dataset import AlzheimerDataset
from model.logistic_regression_model import LogisticRegression


def train():
    dataset = AlzheimerDataset(data_config)
    train_data = dataset('train')
    valid_data = dataset('validation')

    model = LogisticRegression(params_config, experiment_config)

    model.train(train_data['features'], train_data['targets'], valid_data['features'], valid_data['targets'])


def predict():
    # TODO: Set experiment_config.load_model and experiment_config.load_model_epoch
    #       before evaluating on the test dataset
    dataset = AlzheimerDataset(data_config)
    test_data = dataset('test')

    model = LogisticRegression(params_config, experiment_config)
    test_predictions = model.get_model_confidence(test_data['features'])[1, :]

    test_results_df = pd.DataFrame({'ID': test_data['paths'], 'prediction': test_predictions})
    test_results_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    train()

import pandas as pd


def read_dataframe_file(path_to_file: str) -> pd.DataFrame:
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    if path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)
    raise ValueError(
        "Unsupported file format. Only '.csv' and '.pickle' formats are"
        + ' supported.',
    )

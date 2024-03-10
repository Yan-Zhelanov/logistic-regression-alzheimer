from typing import Union

import pandas as pd


def read_dataframe_file(path_to_file: str) -> Union[pd.DataFrame, None]:
    """Reads DataFrame file."""
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    elif path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)
    elif path_to_file.endswith('parquet'):
        return pd.read_parquet(path_to_file)
    else:
        raise ValueError("Unsupported file format")

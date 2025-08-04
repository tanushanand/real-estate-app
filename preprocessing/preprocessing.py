"""
Preprocessing functions for the Real Estate Price Prediction App.
"""

import pandas as pd

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")
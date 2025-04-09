# data_loader.py
"""
Handles loading the dataset.
"""
import pandas as pd
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from an Excel file into a pandas DataFrame.

    Args:
        filepath (str): The path to the Excel file.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For other file reading errors.
    """
    logging.info(f"Attempting to load data from: {filepath}")
    try:
        df = pd.read_excel(filepath)
        logging.info(f"Successfully loaded data with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {filepath}")
        sys.exit(f"Error: Input file not found at {filepath}")
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        sys.exit(f"Error loading data from {filepath}: {e}")
# preprocessing.py
"""
Functions for data preprocessing including text cleaning and airline mention handling.
"""
import pandas as pd
import re
import string
import nltk
# nltk.downloader.DownloadError does not exist, use LookupError for find()
# from nltk.downloader import DownloadError # REMOVE THIS LINE
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import sys # Keep if needed for other potential critical errors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NLTK Setup ---
def setup_nltk():
    """Downloads necessary NLTK data if not found."""
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet'
    }
    for resource_name, resource_path in resources.items():
        try:
            # Check if resource exists using its path
            nltk.data.find(resource_path)
            logging.info(f"NLTK resource '{resource_name}' found.")
        except LookupError: # This is the correct exception when find() fails
            logging.info(f"NLTK resource '{resource_name}' not found. Downloading...")
            try:
                nltk.download(resource_name, quiet=True)
                logging.info(f"NLTK resource '{resource_name}' downloaded successfully.")
            except Exception as e: # Catch potential errors during download (network, etc.)
                logging.error(f"Failed to download NLTK resource '{resource_name}': {e}")
                # Depending on criticality, you might want to exit here:
                # sys.exit(f"Failed to download essential NLTK resource: {resource_name}")
        except Exception as e: # Catch any other unexpected errors during find
            logging.error(f"An unexpected error occurred while checking NLTK resource '{resource_name}': {e}")

    logging.info("NLTK resource check complete.")


# Call setup when module is loaded
setup_nltk()
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


# --- Airline Mention Handling ---

def extract_airlines(text: str, airlines_list: list) -> str | None:
    """
    Extracts airlines mentioned in the text based on a predefined list.

    Args:
        text (str): The input text (tweet).
        airlines_list (list): A list of airline names (lowercase) to search for.

    Returns:
        str | None: A comma-separated string of found airlines, or None if no match.
    """
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    found_airlines = [airline for airline in airlines_list if airline in text_lower]
    return ", ".join(found_airlines) if found_airlines else None

def apply_airline_extraction(df: pd.DataFrame, text_col: str, new_col: str, airlines_list: list) -> pd.DataFrame:
    """Applies the airline extraction function to a DataFrame column."""
    logging.info(f"Extracting airlines from column '{text_col}' into '{new_col}'...")
    df[new_col] = df[text_col].apply(lambda x: extract_airlines(x, airlines_list))
    logging.info("Airline extraction complete.")
    logging.info(f"Value counts for '{new_col}' (before merge):\n{df[new_col].value_counts().head()}")
    return df

def merge_specific_airlines(value: str | None, airlines_to_merge: list, target_airline: str) -> str | None:
    """
    Merges specific airline mentions within a string value into a target airline name.

    Args:
        value (str | None): The comma-separated string of airline mentions (or None).
        airlines_to_merge (list): List of airline names to be merged.
        target_airline (str): The airline name to replace the merged airlines with.

    Returns:
        str | None: The updated comma-separated string of airlines, or None.
    """
    if not isinstance(value, str):
        return value

    airlines = [air.strip() for air in value.split(',')]
    # Replace specified airlines with the target airline
    airlines = [target_airline if air in airlines_to_merge else air for air in airlines]
    # Remove duplicates after merge and sort for consistency
    unique_airlines = sorted(list(set(airlines)))
    return ', '.join(unique_airlines) if unique_airlines else None

def apply_airline_merge(df: pd.DataFrame, col_to_merge: str, airlines_to_merge: list, target_airline: str) -> pd.DataFrame:
    """Applies the airline merging function to a DataFrame column."""
    logging.info(f"Merging airlines {airlines_to_merge} into '{target_airline}' in column '{col_to_merge}'...")
    df[col_to_merge] = df[col_to_merge].apply(lambda x: merge_specific_airlines(x, airlines_to_merge, target_airline))
    logging.info("Airline merging complete.")
    logging.info(f"Value counts for '{col_to_merge}' (after merge):\n{df[col_to_merge].value_counts().head()}")
    return df

def filter_by_mention_count(df: pd.DataFrame, mention_col: str, min_count: int) -> pd.DataFrame:
    """
    Filters the DataFrame to keep only rows where the mentioned airline
    (after potential merging) appears at least min_count times in the dataset.
    Effectively keeps rows with the most frequently mentioned single airlines.

    Args:
        df (pd.DataFrame): The input DataFrame.
        mention_col (str): The name of the column containing airline mentions.
        min_count (int): The minimum count threshold.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    logging.info(f"Filtering DataFrame based on mention count in '{mention_col}' (min count: {min_count})...")
    original_count = len(df)
    airline_counts = df[mention_col].value_counts()
    top_airlines = airline_counts[airline_counts >= min_count].index.tolist()
    filtered_df = df[df[mention_col].isin(top_airlines)].copy() # Use copy to avoid SettingWithCopyWarning later
    filtered_count = len(filtered_df)
    retained_percentage = (filtered_count / original_count) * 100 if original_count > 0 else 0
    logging.info(f"Original dataset size: {original_count}")
    logging.info(f"Filtered dataset size: {filtered_count}")
    logging.info(f"Percentage of data retained: {retained_percentage:.2f}%")
    logging.info(f"Value counts for '{mention_col}' (after filtering):\n{filtered_df[mention_col].value_counts()}")
    return filtered_df


# --- Text Cleaning ---

def clean_text(text: str) -> str:
    """
    Cleans and preprocesses a single string of text (tweet).

    Steps:
    1. Lowercase
    2. Remove mentions (@user) and URLs
    3. Remove punctuation and numbers
    4. Tokenize
    5. Remove stopwords and short words (< 2 chars)
    6. Lemmatize

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # 1) Text Normalization: Convert to lowercase
    text = text.lower()
    # 2) Remove mentions (@user) and URLs
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    # 3) Remove punctuation and numeric characters
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    # 4) Tokenization
    tokens = word_tokenize(text)
    # 5) Remove stopwords and words with length less than 2
    tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 1]
    # 6) Lemmatization
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
    # Rejoin the tokens
    return " ".join(tokens)

def apply_text_cleaning(df: pd.DataFrame, text_col: str, new_col: str) -> pd.DataFrame:
    """Applies the text cleaning function to a DataFrame column."""
    logging.info(f"Applying text cleaning to column '{text_col}', creating '{new_col}'...")
    # Ensure input column is string type, handling potential NaNs
    df[new_col] = df[text_col].astype(str).apply(clean_text)
    logging.info("Text cleaning complete.")
    # Log a sample of cleaned text
    if not df.empty and new_col in df.columns:
         logging.info(f"Sample cleaned text (first 5):\n{df[[text_col, new_col]].head()}")
    return df
# sentiment.py
"""
Handles sentiment analysis using a transformer model.
"""
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_sentiment_model(model_name: str):
    """
    Loads a pre-trained sentiment analysis model and tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the model on Hugging Face Hub.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    logging.info(f"Loading sentiment model and tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logging.info("Sentiment model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {e}")
        raise

def get_sentiment_roberta_binary(text: str, model, tokenizer) -> str:
    """
    Predicts sentiment (Positive/Negative) for a single text using the loaded RoBERTa model.
    Compares only Positive (index 2) and Negative (index 0) scores from the model output.

    Args:
        text (str): The input text (cleaned tweet).
        model: The loaded transformer model.
        tokenizer: The loaded tokenizer.

    Returns:
        str: 'Positive' or 'Negative'.
    """
    if not isinstance(text, str) or not text.strip():
        return 'Negative' # Default for empty or non-string input

    try:
        # Tokenize and truncate if needed
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded_input)
        # The model 'cardiffnlp/twitter-roberta-base-sentiment' outputs: 0: Negative, 1: Neutral, 2: Positive
        scores = softmax(output.logits.numpy()[0])

        # Compare only Positive (index 2) and Negative (index 0)
        if scores[2] > scores[0]:
            return 'Positive'
        else:
            return 'Negative'
    except Exception as e:
        logging.warning(f"Error predicting sentiment for text: '{text[:50]}...'. Error: {e}. Defaulting to Negative.")
        return 'Negative'


def predict_sentiment(df: pd.DataFrame, text_col: str, new_col: str, model, tokenizer) -> pd.DataFrame:
    """
    Applies sentiment prediction to a DataFrame column using the provided model and tokenizer.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_col (str): The name of the column containing the text to analyze.
        new_col (str): The name for the new column containing sentiment predictions.
        model: The loaded transformer model.
        tokenizer: The loaded tokenizer.

    Returns:
        pd.DataFrame: The DataFrame with the added sentiment column.
    """
    logging.info(f"Predicting sentiment for column '{text_col}', creating '{new_col}'...")
    tqdm.pandas(desc="Predicting Sentiment")
    # Ensure the column exists and apply prediction
    if text_col not in df.columns:
         logging.error(f"Text column '{text_col}' not found in DataFrame.")
         raise ValueError(f"Text column '{text_col}' not found in DataFrame.")

    df[new_col] = df[text_col].progress_apply(lambda x: get_sentiment_roberta_binary(x, model, tokenizer))
    logging.info("Sentiment prediction complete.")
    logging.info(f"Value counts for '{new_col}':\n{df[new_col].value_counts()}")
    return df
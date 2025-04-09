# main.py
"""
Main script to run the sentiment analysis workflow.
"""
import pandas as pd
import logging
import argparse # Added for command-line arguments

# Import modules
import config
import data_loader
import preprocessing
import sentiment
import visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(input_filepath):
    """Main function to execute the sentiment analysis pipeline."""
    logging.info("--- Starting Sentiment Analysis Workflow ---")

    # 1. Load Data
    df = data_loader.load_data(input_filepath)

    # 2. Preprocessing - Airline Mentions
    df = preprocessing.apply_airline_extraction(
        df, config.TEXT_COLUMN, config.AIRLINE_MENTION_COLUMN, config.AIRLINES_LIST
    )
    df = preprocessing.apply_airline_merge(
        df, config.AIRLINE_MENTION_COLUMN, config.AIRLINES_TO_MERGE, config.TARGET_MERGED_AIRLINE
    )
    df_filtered = preprocessing.filter_by_mention_count(
        df, config.AIRLINE_MENTION_COLUMN, config.MIN_AIRLINE_MENTION_COUNT
    )

    # 3. Preprocessing - Text Cleaning
    df_processed = preprocessing.apply_text_cleaning(
        df_filtered, config.TEXT_COLUMN, config.CLEAN_TEXT_COLUMN
    )

    # 4. Sentiment Analysis
    # Load model only once
    sentiment_model, sentiment_tokenizer = sentiment.load_sentiment_model(config.SENTIMENT_MODEL_NAME)
    df_final = sentiment.predict_sentiment(
        df_processed, config.CLEAN_TEXT_COLUMN, config.SENTIMENT_COLUMN, sentiment_model, sentiment_tokenizer
    )

    # Optional: Display sample results
    logging.info("--- Sample Final Results (First 10) ---")
    pd.set_option('display.max_colwidth', 100) # Adjust display width for logging
    logging.info(f"\n{df_final[[config.TEXT_COLUMN, config.AIRLINE_MENTION_COLUMN, config.SENTIMENT_COLUMN]].head(10)}")
    pd.reset_option('display.max_colwidth')

    # 5. Visualization
    visualization.plot_sentiment_distribution(
        df_final, config.AIRLINE_MENTION_COLUMN, config.SENTIMENT_COLUMN,
        config.PLOT_COLORS, config.PLOT_FIGURE_SIZE
    )

    logging.info("--- Sentiment Analysis Workflow Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Twitter Sentiment Analysis Pipeline.")
    parser.add_argument(
        "--input",
        type=str,
        default=config.INPUT_FILE_PATH,
        help=f"Path to the input Excel file. Default: {config.INPUT_FILE_PATH}"
    )
    args = parser.parse_args()

    main(args.input)
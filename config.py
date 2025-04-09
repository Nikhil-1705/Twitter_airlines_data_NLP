# config.py
"""
Configuration settings for the sentiment analysis project.
"""

# --- File Paths ---
# INPUT_FILE_PATH = 'C:\\Users\\Nikhil Bhandari\\Desktop\\ML Projects\\Twitter Analysis\\Worksheet in Candidiate Screening Assignment_Associate Data Scientist (002).xlsx'
# Use a relative path or make this configurable if running elsewhere
INPUT_FILE_PATH = 'Worksheet in Candidiate Screening Assignment_Associate Data Scientist (002).xlsx'
# You might want to add OUTPUT_FILE_PATH if saving results

# --- Model Configuration ---
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# --- Data Columns ---
TEXT_COLUMN = 'text'
CLEAN_TEXT_COLUMN = 'clean_text'
AIRLINE_MENTION_COLUMN = 'airlines_mentioned'
SENTIMENT_COLUMN = 'roberta_sentiment'

# --- Preprocessing ---
AIRLINES_LIST = ['southwestair', 'united', 'jetblue', 'americanair', 'usairways', 'virginamerica']
AIRLINES_TO_MERGE = ['usairways', 'americanair']
TARGET_MERGED_AIRLINE = 'usairways' # Keep usairways as the merged name
MIN_AIRLINE_MENTION_COUNT = 100 # Filter out tweets mentioning airlines less than this count

# --- Sentiment Analysis ---
SENTIMENT_LABELS = ['Negative', 'Positive'] # Simplified labels used in the binary prediction

# --- Visualization ---
PLOT_COLORS = 'Set2' # Seaborn color palette
PLOT_FIGURE_SIZE = (6, 6)
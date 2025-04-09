# Twitter Airline Sentiment Analysis

This project analyzes sentiment (Positive/Negative) expressed in tweets directed at major US airlines. It processes a dataset of tweets, cleans the text, performs sentiment prediction using a pre-trained RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment`), and visualizes the sentiment distribution for the most frequently mentioned airlines.

## Features

*   **Data Loading:** Loads tweet data from an Excel file.
*   **Airline Mention Processing:**
    *   Extracts mentions of predefined airlines (`Southwest`, `United`, `JetBlue`, `American Air`, `US Airways`, `Virgin America`) from tweet text.
    *   Merges mentions of 'US Airways' and 'American Air' under a single name ('usairways' by default) to reflect their merger context in the data.
    *   Filters the dataset to focus on airlines mentioned above a certain frequency threshold.
*   **Text Preprocessing:** Cleans tweet text by:
    *   Converting to lowercase.
    *   Removing user mentions (`@username`) and URLs.
    *   Removing punctuation and numbers.
    *   Tokenizing the text.
    *   Removing common English stopwords and very short words.
    *   Lemmatizing words to their base form.
*   **Sentiment Analysis:** Uses a fine-tuned RoBERTa model from Hugging Face (`cardiffnlp/twitter-roberta-base-sentiment`) to classify the cleaned text into 'Positive' or 'Negative' sentiment.
*   **Visualization:** Generates pie charts for each major airline, showing the distribution of positive vs. negative sentiment in the tweets mentioning them.

## Dataset

The project expects an Excel file containing tweet data. By default, it looks for a file named `Worksheet in Candidiate Screening Assignment_Associate Data Scientist (002).xlsx` in the project's root directory. The input file path can be configured in `config.py` or passed as a command-line argument.

The dataset should contain at least a column with the tweet text (expected name: `text`).

## Workflow

The analysis pipeline executes the following steps:

1.  **Load Data:** Reads the specified Excel file into a pandas DataFrame.
2.  **Extract Airlines:** Identifies which airlines are mentioned in each tweet.
3.  **Merge Airlines:** Combines 'US Airways' and 'American Air' mentions.
4.  **Filter Airlines:** Removes tweets associated with airlines mentioned infrequently in the dataset.
5.  **Clean Text:** Applies the preprocessing steps to the tweet text.
6.  **Predict Sentiment:** Uses the loaded RoBERTa model to predict sentiment for the cleaned text.
7.  **Visualize Results:** Displays pie charts illustrating the sentiment breakdown for each included airline.

## Project Structure
Use code with caution.
Markdown
├── config.py # Configuration settings (file paths, model names, parameters)
├── data_loader.py # Function to load data from Excel
├── preprocessing.py # Text cleaning and airline mention processing functions
├── sentiment.py # Sentiment model loading and prediction functions
├── visualization.py # Plotting functions
├── main.py # Main script to orchestrate the workflow
├── requirements.txt # Python package dependencies
└── README.md # This file
└── [Worksheet in Candidiate Screening Assignment_Associate Data Scientist (002).xlsx] # Input data file (place here or specify path)
## Setup and Installation

1.  **Clone or Download:** Get the project files onto your local machine.
2.  **Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **NLTK Data:** The first time you run the script (`main.py`), it will attempt to download the necessary NLTK resources (`punkt`, `stopwords`, `wordnet`) if they are not already present. Ensure you have an internet connection.
5.  **Data File:** Place the input Excel file (`Worksheet in Candidiate Screening Assignment_Associate Data Scientist (002).xlsx`) in the same directory as the scripts, or update the path in `config.py` or use the command-line argument during execution.

## Usage

Run the main script from your terminal:

```bash
python main.py
Use code with caution.
To specify a different input file:
python main.py --input "C:/path/to/your/data_file.xlsx"
Use code with caution.
Bash
The script will output logs to the console detailing the steps, and finally display the sentiment distribution pie charts.
Configuration
Key parameters can be adjusted in the config.py file, including:
INPUT_FILE_PATH: Path to the input data file.
SENTIMENT_MODEL_NAME: The Hugging Face model identifier.
AIRLINES_LIST: The list of airlines to look for.
AIRLINES_TO_MERGE, TARGET_MERGED_AIRLINE: Airlines to combine and their target name.
MIN_AIRLINE_MENTION_COUNT: Threshold for filtering airlines.
Column names (TEXT_COLUMN, SENTIMENT_COLUMN, etc.).
Dependencies
See requirements.txt. Key libraries include:
pandas
openpyxl (for reading Excel)
nltk
torch
transformers
scipy
tqdm
matplotlib
seaborn

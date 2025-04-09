# visualization.py
"""
Functions for visualizing sentiment analysis results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_sentiment_distribution(df: pd.DataFrame, airline_col: str, sentiment_col: str, colors: str | list, fig_size: tuple):
    """
    Generates and displays pie charts showing sentiment distribution for each unique airline.

    Args:
        df (pd.DataFrame): The DataFrame containing airline mentions and sentiments.
        airline_col (str): The name of the column with airline mentions.
        sentiment_col (str): The name of the column with sentiment predictions.
        colors (str | list): Color palette name or list of colors for seaborn.
        fig_size (tuple): Figure size for the plots (width, height).
    """
    logging.info("Generating sentiment distribution plots...")
    unique_airlines = df[airline_col].unique()

    # Set the color palette
    try:
        plot_colors = sns.color_palette(colors)
    except ValueError:
        logging.warning(f"Invalid color palette '{colors}'. Using default.")
        plot_colors = sns.color_palette() # Use default if invalid

    for airline in unique_airlines:
        if not airline: # Skip if airline name is None or empty
            continue

        # Filter the DataFrame for the current airline
        airline_data = df[df[airline_col] == airline]

        # Count sentiment values
        sentiment_counts = airline_data[sentiment_col].value_counts()

        # Skip if there's only one type of sentiment or no data
        if len(sentiment_counts) < 1:
            logging.warning(f"Skipping plot for '{airline}': Not enough sentiment variety or no data.")
            continue

        logging.info(f"Plotting sentiment for: {airline}")
        plt.figure(figsize=fig_size)
        plt.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=plot_colors[:len(sentiment_counts)] # Use only needed colors
        )
        plt.title(f'Sentiment Distribution for {airline}')
        # Consider saving the plot instead of just showing if running as a script
        # plt.savefig(f"sentiment_{airline}.png")
        plt.show()

    logging.info("Plot generation complete.")
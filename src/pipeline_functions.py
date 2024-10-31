# functions.py
import numpy as np
import pandas as pd

# Data Loading


def load_sav_data(file_path):
    """
    Load a file into a DataFrame.

    Parameters:
    file_path (str): The path to the file to load.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    data = pd.read_spss(file_path)
    return data


# Data Cleaning


def relabel_missing_data(df):
    """
    Re-label missing or incorrect inputs as NaN in the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe to process.

    Returns:
    pd.DataFrame: The dataframe with missing or incorrect inputs re-labeled as NaN.
    """
    missing_values = [
        "Missing data (Not Ascertained)",
        "Multiple responses selected in error",
        "Unreadable or Non-conforming numeric response",
        "Missing Data (Filter Missing)",
        "Question Answered in Error (Commission Error)",
        "Inapplicable, coded 2 in EverHadCancer",
        "Inapplicable, this is a Short Form",
        "Missing Data",
        "Inapplicable, coded 1 in EverHadCancer, or this is a Short Form",
        "Inapplicable, coded 1 in BornInUSA",
        "Question answered in error (Commission Error)",
        "Missing Data (Not Ascertained)",
        "Missing Data (Filter Missing), coded -9 in Smoke100",
        "Unreadable or Nonconforming Numeric Response",
        "Inapplicable, coded 0 in FreqGoProvider",
        "Multiple responses selected,in error",
    ]

    df.replace(missing_values, np.nan, inplace=True)

    return df


def get_missing_values(df):
    """
    Calculate the number and percentage of missing values in each column of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame for which to calculate missing values.

    Returns:
    pandas.DataFrame: A DataFrame with two columns:
        - 'n missing': The number of missing values in each column.
        - 'Percentage': The percentage of missing values in each column, sorted in descending order by percentage.
    """
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_results_df = pd.DataFrame(
        {"n missing": missing_count, "Percentage": missing_percentage}
    ).sort_values(by="Percentage", ascending=False)
    return missing_results_df

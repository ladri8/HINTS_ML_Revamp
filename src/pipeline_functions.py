# functions.py
import numpy as np
import pandas as pd

# Data Loading

num_features = ["Age", "BMI"]


def load_sav_data(file_path):
    """
    Load a file into a DataFrame

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


def separate_features(df, num_features):
    """
    Separates the features of a dataframe into numerical and categorical features.

    Parameters:
    df (pd.DataFrame): The dataframe containing the features.
    num_features (list): List of numerical feature names.

    Returns:
    tuple: A tuple containing two lists - numerical features and categorical features.
    """
    all_features = df.columns
    cat_features = [feature for feature in all_features if feature not in num_features]
    return num_features, cat_features


def convert_feature_types(df, num_features, cat_features):
    """
    Converts the data types of features in a dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing the features.
    num_features (list): List of numerical feature names.
    cat_features (list): List of categorical feature names.

    Returns:
    pd.DataFrame: The dataframe with updated feature types.
    """
    # Convert categorical features to 'category' dtype
    for c in cat_features:
        df[c] = df[c].astype("category")

    # Convert numerical features to numeric dtype
    for num_feature in num_features:
        df[num_feature] = pd.to_numeric(df[num_feature], errors="coerce")

    return df.dtypes


def rename_columns(df, old_names, new_names):
    """
    Renames columns in df.

    Parameters:
    df (pandas.DataFrame): The df whose columns are to be renamed.
    old_names (list of str): The current names of the columns to be renamed.
    new_names (list of str): The new names for the columns.

    Returns:
    pandas.DataFrame: The df with renamed columns.

    Example:
    old_names = ['TypeOfAddressC_Selected', 'CellPhone_Yes', 'MedConditions_Diabetes_Yes', ...]
    new_names = ['TypeOfAddressC', 'CellPhone', 'Diabetes', ...]
    """
    name_dict = dict(zip(old_names, new_names))
    df.rename(columns=name_dict, inplace=True)
    return df

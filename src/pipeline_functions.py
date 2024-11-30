# functions.py
import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn

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


def save_in_processed_dir(df, base_dir, file_name="processed_data.csv"):
    """
    Save the given dataframe to the 'processed' directory within the specified base directory.

    Parameters:
    df (pandas.DataFrame): The dataframe to be saved.
    base_dir (str): The base directory where the 'processed' directory is located.
    file_name (str): The name of the file to save the dataframe as.

    Returns:
    None
    """
    # Ensure the directory exists
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Save the dataframe
    df.to_csv(os.path.join(processed_dir, file_name), index=False)


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


def create_dummy_variables(data):
    """
    Create dummy variables for specified columns in the dataframe.

    Parameters:
    data (pd.DataFrame): The input dataframe containing the columns to be converted into dummy variables.

    Returns:
    pd.DataFrame: A dataframe with dummy variables for the specified columns.
    """
    dummy_columns = [
        "TypeOfAddressC",
        "CellPhone",
        "MedConditions_Diabetes",
        "MedConditions_HighBP",
        "MedConditions_HeartCondition",
        "MedConditions_LungDisease",
        "MedConditions_Arthritis",
        "MedConditions_Depression",
        "EverHadCancer",
        "EmotionalSupport",
        "RegularProvider",
        "HealthInsurance",
        "Deaf",
        "Blind",
        "Race_Cat2",
        "Hisp_Cat",
        "BornInUSA",
        "GenderC",
    ]

    DummyVar = pd.get_dummies(data[dummy_columns], drop_first=True)
    return DummyVar


def concatenate_dummies_drop_cols(original_df, dummy_df):
    """
    Concatenate the original dataframe with the dummy variables dataframe and drop specified columns.

    Parameters:
    original_df (pd.DataFrame): The original dataframe.
    dummy_df (pd.DataFrame): The dataframe containing dummy variables.

    Returns:
    pd.DataFrame: The concatenated dataframe with specified columns dropped.
    """
    columns_to_drop = [
        "TypeOfAddressC",
        "CellPhone",
        "MedConditions_Diabetes",
        "MedConditions_HighBP",
        "MedConditions_HeartCondition",
        "MedConditions_LungDisease",
        "MedConditions_Arthritis",
        "MedConditions_Depression",
        "EverHadCancer",
        "EmotionalSupport",
        "RegularProvider",
        "HealthInsurance",
        "Deaf",
        "Blind",
        "Race_Cat2",
        "Hisp_Cat",
        "BornInUSA",
        "GenderC",
    ]

    df = pd.concat([original_df, dummy_df], axis=1).drop(columns_to_drop, axis=1)
    return df


def create_asian_column(df):
    """
    Create a new column 'Asian' in the dataframe based on specific conditions.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the columns to be evaluated.

    Returns:
    pd.DataFrame: The dataframe with the new 'Asian' column added.
    """
    df["Asian"] = np.where(
        (df["Chinese"] == 1)
        | (df["Asian Indian"] == 1)
        | (df["Filipino"] == 1)
        | (df["Japanese"] == 1)
        | (df["Korean"] == 1)
        | (df["OtherAsian"] == 1)
        | (df["Vietnamese"] == 1),
        1,
        0,
    )
    df["Asian"] = df["Asian"].astype(int)
    return df


def create_pacific_islander_column(df):
    """
    Create a new column 'Pacific_Islander' in the dataframe based on specific conditions.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the columns to be evaluated.

    Returns:
    pd.DataFrame: The dataframe with the new 'Pacific_Islander' column added.
    """
    df["Pacific_Islander"] = np.where(
        (df["Samoan"] == 1) | (df["OtherPacificIslander"] == 1), 1, 0
    )
    df["Pacific_Islander"] = df["Pacific_Islander"].astype(int)
    return df


def create_fatalview_column(df):
    """
    Create a new column 'fatalview' in the dataframe based on specific conditions.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the columns to be evaluated.

    Returns:
    pd.DataFrame: The dataframe with the new 'fatalview' column added.
    """
    df["fatalview"] = np.where(
        (df["EverythingCauseCancer"] == 2)
        | (df["EverythingCauseCancer"] == 3)
        | (df["PreventNotPossible"] == 2)
        | (df["PreventNotPossible"] == 3),
        1,
        0,
    )
    return df


def identify_outliers_iqr(df, column):
    """
    Identify the lower and upper bounds for outliers in a DataFrame column using the IQR method.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column (str): The name of the column for which to identify outliers.

    Returns:
    tuple: A tuple containing the lower bound and upper bound for outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


def impute_outliers_with_median(X, columns):
    """
    Impute outliers in specified columns of a DataFrame with the median value of the respective column.

    Parameters:
    X (pd.DataFrame): The input DataFrame containing the data.
    columns (list of str): List of column names in which to impute outliers.

    Returns:
    pd.DataFrame: A new DataFrame with outliers imputed with the median value in the specified columns.

    Notes:
    This function uses the IQR method to identify outliers. It assumes the existence of a function
    `identify_outliers_iqr` that returns the lower and upper bounds for outliers in a given column.
    """
    X = X.copy()
    for column in columns:
        if column not in X.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        lower_bound, upper_bound = identify_outliers_iqr(X, column)
        median_value = X[column].median()
        X[column] = X[column].apply(
            lambda x: median_value if x < lower_bound or x > upper_bound else x
        )
    return X


def create_pipeline(with_imputation=True):
    if with_imputation:
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
        )
    else:
        numeric_transformer = Pipeline(steps=[])
        categorical_transformer = Pipeline(steps=[])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    return preprocessor

import pandas as pd
import numpy as np
import json
import pgeocode
from scipy.stats import zscore

pd.set_option("display.max_columns", None)


class PandasPreprocessor:
    def load_json(self, json_file) -> pd.DataFrame:
        """
        DESCRIPTION
        Load a json file into a dataframe.

        PARAMETERS
        str json_file : Absolute/Relative path of the json file.

        RETURN
        DataFrame object
        """
        with open(json_file) as file:
            dict_json = json.load(file)
        return pd.DataFrame.from_dict(dict_json)

    def preprocess(self, df: pd.DataFrame, save=False) -> pd.DataFrame:
        """
        DESCRIPTION
        Run a series of functions that will perform the first part of
        the preprocessing with Pandas only. (Feature engineering +
        deleting unwanted columns)

        PARAMETERS
        pd.DataFrame df : The DataFrame that is going to be preprocessed.

        RETURN
        DataFrame object
        """
        df = self.get_geo_coordinates(df)
        df = self.delete_columns(df)
        df = self.delete_missing_geo_data(df)
        df = self.remove_outliers(df)

        return df

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DESCRIPTION
        Remove outliers/input errors with zscore.

        PARAMETERS
        pd.DataFrame df : The DataFrame in which you want to convert the booleans.

        RETURN
        DataFrame object
        """
        z_scores = zscore(df["Price"])
        threshold = 3
        outlier_rows = df[(z_scores > threshold)]

        df = df.drop(outlier_rows.index).reset_index()

        return df

    def get_geo_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DESCRIPTION
        Iterate through the postal codes in the dataframe to extract "latitude"
        and "longitude" features.

        PARAMETERS
        pd.DataFrame df : The DataFrame in which the features will be extracted

        RETURN
        DataFrame object
        """
        nomi = pgeocode.Nominatim("be")

        # Create empty lists to store latitude and longitude values
        latitudes = []
        longitudes = []

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            postal_code = row["PostalCode"]
            location_info = nomi.query_postal_code(postal_code)

            # Append latitude and longitude values to the lists
            latitudes.append(location_info.latitude)
            longitudes.append(location_info.longitude)

        # Add latitude and longitude columns to the DataFrame
        df["latitude"] = latitudes
        df["longitude"] = longitudes

        return df

    def delete_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DESCRIPTION
        Delete irrelevant columns like IDs, URLs, columns with an excessive amount
        of null values.

        PARAMETERS
        pd.DataFrame df : The DataFrame in which the columns will be deleted

        RETURN
        DataFrame object
        """
        # First, remove ID and URL
        df = df.drop(columns=["Url", "PropertyId"])

        # Then, remove columns with 50+ % missing values
        for column in df.columns:
            total_values = df[column].size
            missing_values = df[column].isnull().sum()
            missing_values_percent = (missing_values / total_values) * 100

            if missing_values_percent > 50:
                df = df.drop(columns=column)

        return df

    def delete_missing_geo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DESCRIPTION
        There are instances of incorrect postal codes that lead to pgeocode returning None
        for latitude and longitude features. This functions removes these rows.

        PARAMETERS
        pd.DataFrame df : The DataFrame in which the rows will be dropped.

        RETURN
        DataFrame object
        """
        missing_geo_data = df[df["latitude"].isna() | df["longitude"].isna()]
        df = df.drop(missing_geo_data.index)

        return df

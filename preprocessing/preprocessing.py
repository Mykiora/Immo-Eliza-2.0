import pandas as pd
import numpy as np
import json
import pgeocode
from typing import Tuple

pd.set_option("display.max_columns", None)


class Preprocessing:
    def load_json(self, json_file) -> pd.DataFrame:
        """
        DESCRIPTION
        Load a json file into a dataframe.

        PARAMETERS
        str json_file : Absolute/Relative path of the json file.

        RETURN
        Pandas.DataFrame object
        """
        with open(json_file) as file:
            dict_json = json.load(file)

        return pd.DataFrame.from_dict(dict_json)

    def price_range(self, df: pd.DataFrame) -> pd.DataFrame:
        min_price = 90000
        max_price = 1000000

        small_prices = df[df["Price"] < min_price]
        high_prices = df[df["Price"] > max_price]

        df = df.drop(small_prices.index)
        df = df.drop(high_prices.index)

        return df

    def get_geo_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
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
        # First, remove ID and URL
        df = df.drop(columns=["Url", "PropertyId"])

        # Then, remove columns with 50+ % missing values
        for column in df.columns:
            total_values = df[column].size
            missing_values = df[column].isnull().sum()
            missing_values_percent = (missing_values / total_values) * 100

            if missing_values_percent > 50:
                df = df.drop(columns=column)

        # Finally, delete columns that do not correlate with the price
        df = df.drop(columns=["TypeOfProperty", "PostalCode", "TypeOfSale"])

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numerical_columns = df.select_dtypes(include=["int64", "float64"])
        categorical_columns = df.select_dtypes(include="object")

        for column in numerical_columns:
            if df[column].isnull().any():
                df[column] = df[column].fillna(df[column].median())

        for column in categorical_columns:
            if df[column].isnull().any():
                df[column] = df[column].fillna(df[column].mode()[0])

        return df

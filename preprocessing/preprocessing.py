import pandas as pd
import numpy as np
from scipy import stats
import json

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


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

    def price_range(self, df: pd.DataFrame) -> None:
        min_price = 90000
        max_price = 1000000

        small_prices = df[df["Price"] < min_price]
        high_prices = df[df["Price"] > max_price]

        df.drop(small_prices.index, inplace=True)
        df.drop(high_prices.index, inplace=True)

    def delete_columns(self, df: pd.DataFrame) -> None:
        # First, remove ID and URL
        df.drop(columns=["Url", "PropertyId"], inplace=True)

        # Then, remove columns with 50+ % missing values
        for column in df.columns:
            total_values = df[column].size
            missing_values = df[column].isnull().sum()
            missing_values_percent = (missing_values / total_values) * 100

            if missing_values_percent > 50:
                df.drop(columns=column, inplace=True)

        # Finally, delete columns that do not correlate with the price
        df.drop(columns=["TypeOfProperty", "PostalCode", "TypeOfSale"], inplace=True)

    def handle_missing_values(self, df: pd.DataFrame):
        numerical_columns = df.select_dtypes(include=["int64", "float64"])
        categorical_columns = df.select_dtypes(include="object")

        for column in numerical_columns:
            if df[column].isnull().any():
                df[column].fillna(df[column].mean(), inplace=True)

        for column in categorical_columns:
            if df[column].isnull().any():
                df[column].fillna(df[column].mode()[0], inplace=True)

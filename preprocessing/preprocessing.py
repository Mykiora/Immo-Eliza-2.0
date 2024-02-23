import pandas as pd
import numpy as np
import json
import pgeocode
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder
import pickle

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

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.handle_missing_values(df)
        df = self.get_geo_coordinates(df)
        df = self.delete_columns(df)
        df = self.delete_missing_geo_data(df)
        df = self.bool_to_number(df)
        df = self.one_hot_encoding(df)

        return df

    def price_range(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["Price"] >= 90000) & (df["Price"] <= 1000000)]

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

        # Delete columns that do not correlate with the price
        df = df.drop(columns=["TypeOfProperty", "PostalCode", "TypeOfSale"])

        return df

    def delete_missing_geo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_geo_data = df[df["latitude"].isna() | df["longitude"].isna()]
        df = df.drop(missing_geo_data.index)

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

    def bool_to_number(self, df: pd.DataFrame) -> pd.DataFrame:
        bool_columns = df.select_dtypes(include="bool").columns
        df[bool_columns] = df[bool_columns].astype(int)

        return df

    def fit_encoder(self, df: pd.DataFrame) -> None:
        encoder = OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
        )

        categorical_columns = df.select_dtypes(include="object")

        encoder.fit(categorical_columns)

        encoder_file = open("encoder/encoder.obj", "wb")
        pickle.dump(encoder, encoder_file)
        encoder_file.close()

    def one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        encoder_file = open("encoder/encoder.obj", "rb")
        encoder = pickle.load(encoder_file)
        categorical_columns = df.select_dtypes(include="object").reset_index(drop=True)

        encoded_categorical_columns = pd.DataFrame(
            encoder.transform(categorical_columns)
        )

        # cv stands for categorical variable
        df_without_cv = df.drop(columns=categorical_columns.columns).reset_index(
            drop=True
        )

        encoded_df = pd.concat([df_without_cv, encoded_categorical_columns], axis=1)

        return encoded_df

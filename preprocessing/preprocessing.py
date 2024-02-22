import pandas as pd
import numpy as np
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

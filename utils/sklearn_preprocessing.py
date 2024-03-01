import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle


class SklearnPreprocessor:
    def fit_encoder(self, X_train: pd.DataFrame, save=True) -> None:
        """
        DESCRIPTION
        Take the train set (not including the target) and use it to fit
        A pipeline made of SimpleImputer and OneHotEncoding. Optional:
        save the encoder object in a file with pickle.

        PARAMETERS
        pd.DataFrame X_train : The features of the train set.
        bool save : Save the encoder as a file.
        """
        numerical_columns = X_train.select_dtypes(include=["float64", "int64"]).columns
        categorical_columns = X_train.select_dtypes(include=["object", "bool"]).columns

        numerical_pipeline = make_pipeline(SimpleImputer(strategy="median"))
        categorical_pipeline = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
        )

        preprocessor = make_column_transformer(
            (numerical_pipeline, numerical_columns),
            (categorical_pipeline, categorical_columns),
            remainder="passthrough",
        )

        pipeline = make_pipeline(preprocessor)

        pipeline_object = pipeline.fit(X_train)

        # Serialize
        if save:
            with open("utils/encoder.obj", "wb") as file:
                pickle.dump(pipeline_object, file)
        print("Train set successfully fitted !")

    def apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DESCRIPTION
        Import a previously fitted encoder object from a file and apply it
        to some DataFrame (transform).

        PARAMETERS
        pd.DataFrame df : The DataFrame that will be encoded.

        RETURN
        DataFrame object
        """
        with open("utils/encoder.obj", "rb") as file:
            encoder = pickle.load(file)

        encoded_df = pd.DataFrame(encoder.transform(df))
        return encoded_df

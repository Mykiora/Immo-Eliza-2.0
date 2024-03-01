from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


class Model:
    def train(self, X_train, y_train) -> None:
        """
        DESCRIPTION
        Create an instance of XGBoost Regressor and fit it with a DataFrame.
        The model is then serialized and saved into a file with pickle.

        PARAMETERS
        pd.DataFrame X_train : A DataFrame containing data's features.
        pd.Series y_train : A Series containing the target variable.

        RETURN
        None
        """
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train, verbose=True)

        with open("model/model.obj", "wb") as xgb_file:
            pickle.dump(xgb, xgb_file)

    def test(self, y_test, y_predictions) -> None:
        """
        DESCRIPTION
        Evaluate the model by calculating the RMSE and R².

        PARAMETERS
        pd.Series y_test : A series containing the ACTUAL property price from the test set.
        pd.Series y_predictions : A series containing the prices predicted by the model.

        RETURN
        None
        """
        print(f"RMSE : {np.sqrt(mean_absolute_error(y_predictions, y_test))}")
        print(f"R² : {r2_score(y_predictions, y_test)}")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        DESCRIPTION
        Import the XGBoost Regressor from a file and predict prices.

        PARAMETERS
        pd.DataFrame X : A DataFrame containing data's features (without the target).

        RETURN
        DataFrame object
        """
        with open("model/model.obj", "rb") as xgb_file:
            xgb = pickle.load(xgb_file)

        predictions = pd.DataFrame(xgb.predict(X))

        return predictions

    def save_results(self, y_test, y_predictions) -> None:
        """
        DESCRIPTION
        Save the Results DataFrame obtained from predict() in a csv file.

        PARAMETERS
        pd.Series y_test : A series containing the ACTUAL property price from the test set.
        pd.Series y_predictions : A series containing the prices predicted by the model.

        RETURN
        None
        """
        result = pd.concat(
            [y_test.reset_index(drop=True), y_predictions.reset_index(drop=True)],
            axis=1,
        )

        result = result.rename(columns={0: "Prediction"})

        result.to_csv("results/results.csv")

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score


class Model:
    def test(self, X_train, y_train, X_test, y_test) -> None:
        """
        DESCRIPTION
        This function uses the train set to fit a model, make predictions,
        and evaluate the results by testing the model on the test set.

        PARAMETERS
        pd.DataFrame X_train, y_train : Train set DataFrame
        pd.DataFrame X_test, y_test   : Test set DataFrame

        RETURN
        None
        """
        model = XGBRegressor()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Evaluation
        print(f"RMSE : {np.sqrt(mean_squared_error(y_test, predictions))}")
        print(f"R² : {r2_score(y_test, predictions)}")

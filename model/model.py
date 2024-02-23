from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import pickle


class Model:
    def train(self, X_train, y_train):
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train, verbose=True)

        with open("model/model.obj", "wb") as xgb_file:
            pickle.dump(xgb, xgb_file)

    def test(self, y_test, y_predictions):
        print(f"RMSE : {np.sqrt(mean_absolute_error(y_predictions, y_test))}")

    def predict(self, X: pd.DataFrame):
        with open("model.obj", "rb") as xgb_file:
            xgb = pickle.load(xgb_file)

        predictions = xgb.predict(X)

        return predictions

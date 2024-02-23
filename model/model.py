from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


class Model:
    def train(self, X_train, y_train):
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train, verbose=True)

        with open("model/model.obj", "wb") as xgb_file:
            pickle.dump(xgb, xgb_file)

    def test(self, y_test, y_predictions):
        print(f"RMSE : {np.sqrt(mean_absolute_error(y_predictions, y_test))}")
        print(f"R² : {r2_score(y_predictions, y_test)}")

    def predict(self, X: pd.DataFrame):
        with open("model/model.obj", "rb") as xgb_file:
            xgb = pickle.load(xgb_file)

        predictions = pd.DataFrame(xgb.predict(X))

        return predictions

    def save_results(self, y_test, y_predictions):
        result = pd.concat(
            [y_test.reset_index(drop=True), y_predictions.reset_index(drop=True)],
            axis=1,
        )

        result = result.rename(columns={0: "Prediction"})

        result.to_csv("results/results.csv")

from utils.pandas_preprocessing import PandasPreprocessor
from utils.model import Model
from utils.sklearn_preprocessing import SklearnPreprocessor
import pandas as pd
import pickle

# Preprocessors instances
prep_pd = PandasPreprocessor()
prep_sklearn = SklearnPreprocessor()

# Load data
# train, test = prep_pd.load_json("data/train.json"), prep_pd.load_json("data/test.json")
with open("utils/train_pandas_prep_df.obj", "rb") as file:
    train = pickle.load(file)

with open("utils/test_pandas_prep_df.obj", "rb") as file:
    test = pickle.load(file)

# Preprocessing - first part (with pandas)
"""print("Preprocessing of the train set with pandas begins...")
train = prep_pd.preprocess(train)
print("I'm done with the train set !")
print("Now moving to the test set...")
test = prep_pd.preprocess(test)
print("Done !")"""

# Split features and target variables
X_train, y_train = train.drop("Price", axis=1), train["Price"]
X_test, y_test = test.drop("Price", axis=1), test["Price"]

# Preprocessing - second part (sklearn)
# Fit ↓↓↓ is optional
prep_sklearn.fit_encoder(X_train)
X_train_preprocessed = prep_sklearn.apply_encoding(X_train)
X_test_preprocessed = prep_sklearn.apply_encoding(X_test)

# Test Model
model = Model()
model.test(X_train_preprocessed, y_train, X_test_preprocessed, y_test)

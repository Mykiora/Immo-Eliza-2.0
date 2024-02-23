from preprocessing.preprocessing import Preprocessing
from model.model import Model

prep = Preprocessing()

# Load data
train = prep.load_json("data/train.json")
test = prep.load_json("data/test.json")

# Preprocessing
train = prep.preprocess(train)
test = prep.preprocess(test)

# Split features and target variables
X_train = train.drop("Price", axis=1)
y_train = train["Price"]

X_test = test.drop("Price", axis=1)
y_test = test["Price"]

# Model instance
xgb = Model()

# Train (optional)
# xgb.train(X_train, y_train)

# Predict
predictions = xgb.predict(X_test)

# Test model
xgb.test(y_test, predictions)

# Save result
# xgb.save_results(y_test, predictions)

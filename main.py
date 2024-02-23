from preprocessing.preprocessing import Preprocessing
from model.model import Model
import pickle

prep = Preprocessing()

# Load data
train = prep.load_json("data/train.json")
test = prep.load_json("data/test.json")

# Filtering price range
train = prep.price_range(train)

# Preprocessing
train = prep.preprocess(train)
test = prep.preprocess(test)

# Split features and target variables
X_train = train.drop("Price", axis=1)
y_train = train["Price"]

X_test = test.drop("Price", axis=1)
y_test = test["Price"]

# Train model
model = Model()
model.train(X_train, y_train)

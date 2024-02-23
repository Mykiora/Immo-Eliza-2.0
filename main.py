from preprocessing.preprocessing import Preprocessing
import pickle

prep = Preprocessing()

# Load data
train = prep.load_json("data/train.json")
test = prep.load_json("data/test.json")

# Filtering price range
train = prep.price_range(train)

# Train and target variables
X_train = train.drop("Price", axis=1)
y_train = train["Price"]

X_test = test.drop("Price", axis=1)
y_test = test["Price"]

# Preprocessing
X_train = prep.preprocess(X_train)
X_test = prep.preprocess(X_test)

print(X_train.shape)
print(X_test.shape)

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
X_train = prep.handle_missing_values(X_train)
X_train = prep.get_geo_coordinates(X_train)
X_train = prep.delete_columns(X_train)
X_train = prep.delete_missing_geo_data(X_train)
X_train = prep.bool_to_number(X_train)
X_train = prep.one_hot_encoding(X_train)
print(X_train.head(3))

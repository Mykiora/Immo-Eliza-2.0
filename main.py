from preprocessing.preprocessing import Preprocessing
import pgeocode

prep = Preprocessing()

train = prep.load_json("data/train.json")
test = prep.load_json("data/test.json")

train = prep.price_range(train)
train = prep.handle_missing_values(train)
prep.get_geo_coordinates(train)
train = prep.delete_columns(train)
print(train.head(3))

from preprocessing.preprocessing import Preprocessing

prep = Preprocessing()

train = prep.load_json("data/train.json")
test = prep.load_json("data/test.json")
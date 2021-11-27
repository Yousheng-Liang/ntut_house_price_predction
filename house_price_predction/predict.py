import csv
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, scale
from keras.models import Sequential
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plot
from sklearn.utils import shuffle

# Load Dataset
train_path = "dataset/ntut-ml-regression-2021/train-v3.csv"
train_file = pd.read_csv(train_path)
train_file = shuffle(train_file)

# Load Test Dataset
test_path = "dataset/ntut-ml-regression-2021/test-v3.csv"
test_file = pd.read_csv(test_path)

# Split Dataset into input part and target part
x_train = train_file.drop(["id", "price"], axis=1).values
y_train = train_file.price.values
x_test = test_file.drop(["id"], axis=1).values

# Scale Data
scaler = StandardScaler().fit(x_train)
x_train = scale(x_train)
x_test = scaler.transform(x_test)

# Load Model
myModel = load_model("myModel.h5")

# myModel.summary()

# Predict the Result and Save
y_test = myModel.predict(x_test)

with open("result.csv", "w", newline="") as result :
    writer = csv.writer(result)
    writer.writerow(["id", "price"])

    for i in range(y_test.shape[0]):
        writer.writerow([(i+1), y_test[i][0]])
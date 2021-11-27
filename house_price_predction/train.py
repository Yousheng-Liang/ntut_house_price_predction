from tensorflow.keras.models import save_model
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, scale
from keras.models import Sequential
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plot
from sklearn.utils import shuffle

# Load Dataset
train_path = "dataset/ntut-ml-regression-2021/train-v3.csv"
train_file = pd.read_csv(train_path)
train_file = shuffle(train_file)

#print(train_file)

valid_path = "dataset/ntut-ml-regression-2021/valid-v3.csv"
valid_file = pd.read_csv(valid_path)

test_path = "dataset/ntut-ml-regression-2021/test-v3.csv"
test_file = pd.read_csv(test_path)

# Split Dataset into input part and target part
x_train = train_file.drop(["id", "price"], axis=1).values
y_train = train_file.price.values

x_valid = valid_file.drop(["id", "price"], axis=1).values
y_valid = valid_file.price.values

x_test = test_file.drop(["id"], axis=1).values

# Scale Data
scaler = StandardScaler().fit(x_train)
x_train = scale(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

# Create Model
myModel = Sequential()
myModel.add(Dense(500, input_dim=x_train.shape[1], activation="relu", kernel_initializer="normal"))
myModel.add(Dropout(0.25))
myModel.add(Dense(400, input_dim=x_train.shape[1], activation="relu", kernel_initializer="normal"))
myModel.add(Dropout(0.2))
myModel.add(Dense(150, activation="relu", kernel_initializer="normal"))
myModel.add(Dropout(0.2))
myModel.add(Dense(50, activation="relu", kernel_initializer="normal"))
# myModel.add(Dropout(0.2))
myModel.add(Dense(20, activation="relu", kernel_initializer="normal"))
myModel.add(Dense(1, activation='linear'))

opt = Adam(learning_rate=0.0004)
myModel.compile(optimizer=opt, loss="MAE")

# myModel.summary()

# Setting Hyper parameters
epoches = 3000
batch_size = 2048

# Start Training Model
myModel.fit(x_train, y_train,batch_size=batch_size, epochs=epoches, validation_data=(x_valid, y_valid))
history = pd.DataFrame(myModel.history.history)
plot.plot(history)
plot.legend(["loss", "val_loss"])
plot.show()

# Save the Model
save_model(myModel, "myModel.h5")


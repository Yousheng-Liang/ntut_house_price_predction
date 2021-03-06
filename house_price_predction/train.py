import pandas as pd  # 用來處理資料，本專案用於處理csv的讀取及篩選
from tensorflow.keras.optimizers import Adam  # Adam Optimizer
from sklearn.preprocessing import StandardScaler, scale  # 用以進行資料標準化
from keras.models import Sequential  # 可將各層模型連接起來
from keras.layers import Dense, Dropout  # 用於Hidden Layer 內
from tensorflow.keras.models import save_model  # 可將訓練後的模型儲存
from matplotlib import pyplot as plot  # 可將訓練結果可視化，方便觀察模型表現
from sklearn.utils import shuffle  # 用以將讀入的csv內容打亂

# Load Dataset
train_path = "dataset/ntut-ml-regression-2021/train-v3.csv "  # Train Dataset的path
train_file = pd.read_csv(train_path)  # 將csv檔內容讀入變數
train_file = shuffle(train_file)  # 將輸入打亂，以便獲得更好訓練效果

valid_path = "dataset/ntut-ml-regression-2021/valid-v3.csv"   # Valid Dataset的path
valid_file = pd.read_csv(valid_path)  # 將csv檔內容讀入變數

test_path = "dataset/ntut-ml-regression-2021/test-v3.csv"      # Test Dataset的path
test_file = pd.read_csv(test_path)     # 將csv檔內容讀入變數

# Split Dataset into input part and target part
x_train = train_file.drop(["id", "price"], axis=1).values  # 丟棄id及price欄位
y_train = train_file.price.values  # 將price欄位的值存入變數做為traget

x_valid = valid_file.drop(["id", "price"], axis=1).values  # 丟棄id及price欄位
y_valid = valid_file.price.values  # 將price欄位的值存入變數做為traget

x_test = test_file.drop(["id"], axis=1).values  # 丟棄id欄位

# Scale Data
scaler = StandardScaler().fit(x_train)
x_train = scale(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

# Create Model
myModel = Sequential()  # Sequential會自動將各層連接起來
# 加入神經元個數為500的Dense層，並設定輸入的shape。由於Sequential會自動連接，故後面的都不需設定shape
myModel.add(Dense(500, input_dim=x_train.shape[1], activation="relu", kernel_initializer="normal"))
# 加入25%的Dropout
myModel.add(Dropout(0.25))
# 加入神經元個數為400的Dense層
myModel.add(Dense(400, activation="relu", kernel_initializer="normal"))
# 加入20%的Dropout
myModel.add(Dropout(0.2))
# 加入神經元個數為150的Dense層
myModel.add(Dense(150, activation="relu", kernel_initializer="normal"))
# 加入20%的Dropout
myModel.add(Dropout(0.2))
# 加入神經元個數為50的Dense層
myModel.add(Dense(50, activation="relu", kernel_initializer="normal"))
# 加入神經元個數為20的Dense層
myModel.add(Dense(20, activation="relu", kernel_initializer="normal"))
# 建立output layer
myModel.add(Dense(1, activation='linear'))
# 設定Adam的learning_rate為0.0004
opt = Adam(learning_rate=0.0004)
myModel.compile(optimizer=opt, loss="MAE")

# myModel.summary()

# Setting Hyper parameters
epoches = 3000
batch_size = 2048

# Start Training Model
myModel.fit(x_train, y_train,batch_size=batch_size, epochs=epoches, validation_data=(x_valid, y_valid))
history = pd.DataFrame(myModel.history.history)  # 將訓練結果存入變數
plot.plot(history)  # 繪出訓練結果(loss和val_loss)
plot.legend(["loss", "val_loss"])
plot.show()

# Save the Model
save_model(myModel, "myModel.h5")


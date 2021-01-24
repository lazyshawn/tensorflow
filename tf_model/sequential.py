# ====== import 相关模块 ======
# tf模块
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# 其他模块
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ====== 指定数据集 ======
# 训练集的输入特征和标签
directory = "../DataProcess/dataSet/"
column_names = ['time', 'x', 'y', 'theta', 'u', 'v', 'omega', 'label_u', 'label_v', 'label_omega']
value = []
data, label = [], []
with open(directory+"dataSet.txt", "r") as file:
    for line in file.readlines():
        #  value.append(line.split()[1:10])
        data.append(line.split()[1:7])
        label.append(line.split()[7:10])

train_data = pd.DataFrame(data).astype(float)
train_data.columns = column_names[1:7]
train_label = pd.DataFrame(label).astype(float)
train_label.columns = column_names[7:10]

print(train_data)
print(train_label)

# 测试集的输入特征和标签
value = []
data, label = [], []
with open(directory+"test_data.txt", "r") as file:
    for line in file.readlines():
        data.append(line.split()[1:7])
        label.append(line.split()[7:10])

test_data = pd.DataFrame(data).astype(float)
test_data.columns = column_names[1:7]
test_label = pd.DataFrame(label).astype(float)
test_label.columns = column_names[7:10]


# 数据归一化
train_stats= train_data.describe().transpose()
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_data)
normed_test_data = norm(test_data)


# ====== 构建模型 ======
def build_model():
  model = keras.Sequential([
    layers.Dense(256, activation='tanh', input_shape=[len(train_data.keys())]),
    layers.Dense(256, activation='tanh'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics='mse')
  return model

model = build_model()


# ====== 训练模型 ======
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_label,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot(), early_stop])

# ====== 训练结果 ======
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  #  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)


# ====== 预测 ======
test_predictions = model.predict(normed_test_data)
error = (test_predictions - test_label).transpose()
error = error.describe().transpose()['std']
print(error)
plt.hist(error, bins = 100)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()



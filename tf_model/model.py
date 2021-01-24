# ======================================
# ==>> import 相关模块
# ======================================
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
from sklearn.preprocessing import StandardScaler


# ======================================
# ==>> 指定数据集
# ======================================
# 训练集的输入特征和标签
directory = "../DataProcess/dataSet/"
scaler=StandardScaler()
train_data, train_label = [], []
with open(directory+"dataSet.txt", "r") as file:
    for line in file.readlines():
        value = line.split()
        train_data.append(value[1:7])
        train_label.append(value[7:10])

train_data = np.array(train_data).astype(np.float)
train_label = np.array(train_label).astype(np.float)

# 测试集的输入特征和标签
time, test_data, test_label = [], [], []
with open(directory+"test_data.txt", "r") as file:
    for line in file.readlines():
        value = line.split()
        time.append(value[0])
        test_data.append(value[1:7])
        test_label.append(value[7:10])

test_data = np.array(test_data).astype(np.float)
test_label = np.array(test_label).astype(np.float)

# 归一化
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# ======================================
# ==>> 搭建网络结构
# ======================================
# 逐层描述每层网络, 相当于走了一遍前向传播
model = keras.Sequential([
    layers.Dense(64, activation='relu',input_shape=train_data.shape[1:]),
    layers.Dense(64, activation='relu'),
    layers.Dense(3)
    ])


# ======================================
# ==>> 在 compile() 中配置训练方法
# ======================================
# 优化器、损失函数、评测指标
model.compile(loss='mse',
        optimizer=tf.keras.optimizers.RMSprop(0.0005),
        metrics='mse')


# ======================================
# ==>> 在 fit() 中执行训练过程
# ======================================
# 训练集、测试集的输入特征和标签，batch、epoch
EPOCHS = 1000
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
        train_data_scaled, train_label,
        epochs=EPOCHS, validation_split=0.2,
        callbacks=early_stop
        )


# ======================================
# ==>> 用 summary() 打印出网络的结构和参数统计
# ======================================
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
  plt.ylim([0,20])
  plt.legend()

plot_history(history)
plt.show()


# ======================================
# ==>> 后处理
# ======================================
# 预测
test_predictions = model.predict(test_data_scaled)
error = test_predictions - test_label
error = np.linalg.norm(error, axis=1)
truth = np.linalg.norm(test_label, axis=1)
error = error / truth

plt.hist(error, bins = 1025)
plt.xlabel("Prediction Error [MSE]")
_ = plt.ylabel("Count")
plt.show()


plt.figure()
plt.plot(time, test_predictions[:,0])
plt.plot(time, test_data[:,3])
plt.plot(time, test_predictions[:,1])
plt.plot(time, test_data[:,4])
plt.plot(time, test_predictions[:,2])
plt.plot(time, test_data[:,5])
plt.show()




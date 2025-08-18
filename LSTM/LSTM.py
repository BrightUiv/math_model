import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. 数据准备
time = np.arange(0, 1000, 0.1)
# 混合两个频率的正弦波并加入噪声
amplitude = np.sin(time * 0.5) + np.sin(time * 0.1) + np.random.normal(scale=0.05, size=len(time))

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(amplitude.reshape(-1, 1))

# 分割训练集和测试集
train_size = int(len(data_scaled) * 0.80)
train_data, test_data = data_scaled[0:train_size,:], data_scaled[train_size:len(data_scaled),:]

def create_multistep_dataset(dataset, time_step_in, time_step_out):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step_in - time_step_out + 1):
        a = dataset[i:(i + time_step_in), 0]
        dataX.append(a)
        b = dataset[(i + time_step_in):(i + time_step_in + time_step_out), 0]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

# 定义输入和输出的时间步长
n_steps_in = 60
n_steps_out = 10

# 创建训练集和测试集
X_train, y_train = create_multistep_dataset(train_data, n_steps_in, n_steps_out)
X_test, y_test = create_multistep_dataset(test_data, n_steps_in, n_steps_out)

# Reshape输入为3D张量 [样本数, 时间步长, 特征数]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"训练集输入形状: {X_train.shape}")
print(f"训练集输出形状: {y_train.shape}")


# --- 2. 构建优化后的 "多对多" LSTM 模型 ---

model = Sequential()

# **优化点 1: 堆叠LSTM层**
# 第一层LSTM。units增加到128，return_sequences=True 以便将序列输出给下一层
model.add(LSTM(units=128, return_sequences=True, input_shape=(n_steps_in, 1)))
# **优化点 2: 添加Dropout层防止过拟合**
model.add(Dropout(0.2))

# 第二层LSTM
model.add(LSTM(units=64, return_sequences=False)) # 最后一层LSTM，无需返回序列
model.add(Dropout(0.2))

# 输出层
model.add(Dense(units=n_steps_out))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 打印模型结构
model.summary()


# --- 3. 训练模型 (使用早停法) ---
print("\n开始训练模型...")
# **优化点 3: 定义EarlyStopping回调**
# monitor='val_loss': 监控验证集的损失
# patience=10: 如果验证集损失在10个epoch内没有改善，则停止训练
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

# 训练模型，增加epochs，并加入回调
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50, # 可以设置一个较大的值，EarlyStopping会自动找到最佳点
    batch_size=64,
    verbose=1,
    callbacks=[early_stopping] # 将回调传入训练过程
)
print("模型训练完成。")


# --- 4. 评估模型并输出损失 ---
# 在测试集上评估模型，获取最终的损失值
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"\n模型在测试集上的最终损失 (MSE): {test_loss:.6f}")

# 可视化训练过程中的损失变化
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


# --- 5. 可视化预测结果 ---
# 在测试集上进行预测
test_predict = model.predict(X_test)

# 反归一化
test_predict = scaler.inverse_transform(test_predict)
y_test_orig = scaler.inverse_transform(y_test)
x_test_orig = scaler.inverse_transform(X_test[:,:,0])

# 可视化一个预测样本
plt.figure(figsize=(14, 7))
plt.style.use('seaborn-v0_8-whitegrid')

sample_index = 100
plt.plot(np.arange(n_steps_in), x_test_orig[sample_index], label='Historical Input Data', color='blue')
plt.plot(np.arange(n_steps_in, n_steps_in + n_steps_out), y_test_orig[sample_index], label='Actual Future Data', color='green', marker='o')
plt.plot(np.arange(n_steps_in, n_steps_in + n_steps_out), test_predict[sample_index], label='Predicted Future Data', color='red', linestyle='--', marker='x')

plt.title(f'Optimized Multi-Step Prediction (Sample {sample_index})', fontsize=16)
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.legend(fontsize=12)
plt.axvline(x=n_steps_in - 0.5, color='gray', linestyle=':')
plt.show()


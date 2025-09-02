import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import re
import os  # 导入os模块

# --- 1. 准备数据和设置参数 ---

# 设置超参数
vocab_size = 10000
max_length = 200
embedding_dim = 128
model_path = 'sentiment_lstm_model.keras'  # 将模型路径定义在前面

# 加载 IMDB 数据集
# 注意：这部分数据主要用于训练，如果只是预测，其实可以跳过
# 但为了获取word_index，我们还是在这里加载它
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 填充序列 (仅为训练需要)
x_train = pad_sequences(x_train, maxlen=max_length, padding='pre')
x_test = pad_sequences(x_test, maxlen=max_length, padding='pre')

# --- 2. 加载或训练模型 ---

# 检查模型文件是否已经存在
if os.path.exists(model_path):
    print(f"发现已存在的模型，从 {model_path} 加载...")
    model = load_model(model_path)
    print("模型加载成功。")
else:
    # 如果模型不存在，则构建、训练并保存它
    print("未发现模型，开始构建和训练新模型...")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    print("\n开始训练模型...")
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2
    )
    print("模型训练完成。")

    print(f"\n将新训练的模型保存到: {model_path}")
    model.save(model_path)
    print("模型保存成功。")

# --- 3. 创建预测函数 ---
# 无论前面是加载还是新训练，到这里我们都有一个可用的 'model' 对象

# 获取 IMDB 词汇表索引
word_index = imdb.get_word_index()
index_from = 3
word_to_index = {k: (v + index_from) for k, v in word_index.items()}
word_to_index["<PAD>"] = 0
word_to_index["<START>"] = 1
word_to_index["<UNK>"] = 2


# (请在这里粘贴您之前已经修正好的 predict_sentiment 函数)
def predict_sentiment(text):
    # (这里是完整的、修正了索引错误的predict_sentiment函数)
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    encoded_words = []
    for word in words:
        encoded_word = word_to_index.get(word, 2)
        if encoded_word < vocab_size:
            encoded_words.append(encoded_word)
        else:
            encoded_words.append(2)

    padded_sequence = pad_sequences([encoded_words], maxlen=max_length, padding='pre')
    prediction = model.predict(padded_sequence, verbose=0)  # 注意：这里使用 model, 而不是 loaded_model
    probability = prediction[0][0]
    print(f"预测概率: {probability:.4f}")

    if probability > 0.5:
        return "正面 (Positive)"
    else:
        return "负面 (Negative)"


# --- 4. 对手动输入的语句进行测试 ---
# (这部分代码无需改动)
print("\n--- 开始手动测试 ---")
print("现在你可以输入自己的英文句子进行测试了 (输入 'quit' 退出):")
while True:
    user_input = input("请输入一句英文影评: ")
    if user_input.lower() == 'quit':
        break
    result = predict_sentiment(user_input)
    print(f"-> 模型判断为: {result}\n")
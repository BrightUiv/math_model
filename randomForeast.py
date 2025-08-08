# 1. 导入必要的库
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归器
from sklearn.metrics import mean_squared_error, r2_score # 导入评估指标
from sklearn.datasets import fetch_california_housing # 导入数据集

# 2. 加载数据
# Scikit-learn 内置了加州住房数据集
housing = fetch_california_housing()
X = housing.data  # 特征 (收入、房屋年龄、房间数等)
y = housing.target # 目标 (房价中位数)

print("数据特征的形状:", X.shape)
# 输出: 数据特征的形状: (20640, 8)  (代表有20640个样本，每个样本8个特征)

# 3. 将数据划分为训练集和测试集
# 80% 的数据用于训练，20% 的数据用于测试
# random_state 是一个随机种子，确保每次划分结果都一样，方便复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("训练集大小:", X_train.shape[0])
print("测试集大小:", X_test.shape[0])

# 4. 创建并训练随机森林回归模型
# 这里是关键步骤！
# n_estimators=100 表示我们要构建一个包含100棵决策树的森林。
# random_state=42 同样是为了结果可复现。
# n_jobs=-1 表示使用所有可用的CPU核心来并行训练，可以加快速度。
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 使用 .fit() 方法在训练数据上训练模型
# 在这一步，Scikit-learn 会自动完成我们之前讨论的所有复杂工作：
# - Bagging：有放回地随机抽取样本，构建100个不同的训练集
# - 特征随机化：在每个节点分裂时，随机选择部分特征
# - 构建100棵决策树
print("\n开始训练随机森林模型...")
rf_regressor.fit(X_train, y_train)
print("模型训练完成！")

# 5. 使用训练好的模型进行预测
# 我们用模型对“它没见过”的测试集数据(X_test)进行预测
print("\n在测试集上进行预测...")
y_pred = rf_regressor.predict(X_test)

# -------------------------------------------------------------------------
# 重点：上面这行 .predict() 就是你问题的答案！
# 它在内部自动完成了对森林中100棵树的预测，并计算了这些预测值的平均数，
# 最终返回那个平均值作为最终的预测结果(y_pred)。
# 我们完全不需要手动编写循环和求平均值的代码。
# -------------------------------------------------------------------------

# 6. 评估模型性能
# 我们将模型的预测值(y_pred)与真实的房价(y_test)进行比较

# 均方误差 (Mean Squared Error, MSE): 预测值与真实值之差的平方的平均值。越小越好。
mse = mean_squared_error(y_test, y_pred)

# R^2 分数 (R-squared): 决定系数，表示模型对数据变化的解释程度。值在0到1之间，越接近1越好。
r2 = r2_score(y_test, y_pred)

print("\n模型性能评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R^2 分数: {r2:.4f}")

# 我们可以看一个预测样本
print("\n--- 预测示例 ---")
print(f"第一个测试样本的真实房价: {y_test[0]:.2f}")
print(f"模型预测的房价: {y_pred[0]:.2f}")
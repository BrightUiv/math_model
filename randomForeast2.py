# =============================================================================
#  方便调参的随机森林代码模板 (分类与回归)
# =============================================================================

# 1. 导入所需库
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
import time

# =============================================================================
# 2. 全局控制与参数区
# =============================================================================

# 用于快速切换分类和回归任务
PROBLEM_TYPE = 'classification'
# PROBLEM_TYPE = 'regression'

# 定义你想要测试的参数网格
# GridSearchCV 将会测试下面所有参数的组合
param_grid = {
    # n_estimators: 森林中树的数量。通常越多越好，但计算成本会增加。
    'n_estimators': [100, 200, 300],

    # max_depth: 树的最大深度。用于防止过拟合。
    'max_depth': [None, 10, 20],

    # max_features: 寻找最佳分裂时要考虑的特征数量。是候选特征的数量，分裂的时候还是只用一个特征。
    'max_features': ['sqrt', 'log2', 0.8],
    # sqrt表示选择所有特征数量的根号数量

    # min_samples_leaf: 一个叶子节点必须包含的最小样本数。用于防止过拟合。
    'min_samples_leaf': [1, 5, 10],

    # min_samples_split: 分裂一个内部节点所需的最小样本数。
    'min_samples_split': [2, 10]
}

# =============================================================================
# 3. 数据加载与准备
# =============================================================================
print(f"--- 任务开始: {PROBLEM_TYPE} ---")

if PROBLEM_TYPE == 'classification':
    # 加载经典的鸢尾花(iris)分类数据集
    data = load_iris()
    X, y = data.data, data.target
    print("加载数据集: Iris (鸢尾花分类)")
else:  # regression
    # 加载加州房价(california housing)回归数据集
    data = fetch_california_housing()
    X, y = data.data, data.target
    print("加载数据集: California Housing (加州房价回归)")

# 将数据划分为训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape[0]} 个样本  {X_train.shape[1]}个特征")
print(f"测试集大小: {X_test.shape[0]} 个样本   {X_test.shape[1]}个特征")
print("-" * 30)

# =============================================================================
# 4. 模型选择与评估指标
# =============================================================================
if PROBLEM_TYPE == 'classification':
    model = RandomForestClassifier(random_state=42)
    scoring_metric = 'accuracy'  # 使用准确率作为交叉验证的评估指标
    print("模型: RandomForestClassifier")
else:  # regression
    model = RandomForestRegressor(random_state=42)
    scoring_metric = 'r2'  # 使用R²分数作为交叉验证的评估指标
    print("模型: RandomForestRegressor")

print(f"交叉验证评估指标: {scoring_metric}")
print("-" * 30)

# =============================================================================
# 5. 使用GridSearchCV进行自动调参
# =============================================================================
print("开始使用GridSearchCV进行自动调参...")
print("定义的参数网格:")
print(param_grid)

# cv=5 表示5折交叉验证, n_jobs=-1 表示使用所有可用的CPU核心并行计算
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring=scoring_metric)

start_time = time.time()
grid_search.fit(X_train, y_train)  # 在训练集上进行搜索
end_time = time.time()

print(f"\nGridSearchCV 完成! 总耗时: {end_time - start_time:.2f} 秒")
print("-" * 30)

# =============================================================================
# 6. 输出最佳结果
# =============================================================================
print("--- 调参结果 ---")
# 输出在交叉验证中找到的最佳参数组合
print(f"最佳参数组合 (Best Parameters): {grid_search.best_params_}")

# 输出在交叉验证中得到的最佳评估分数
print(f"最佳交叉验证分数 (Best Score): {grid_search.best_score_:.4f}")

# 获取训练好的最佳模型
best_model = grid_search.best_estimator_
print("-" * 30)

# =============================================================================
# 7. 在测试集上评估最终模型
# =============================================================================
print("--- 在独立的测试集上评估最终模型性能 ---")

y_pred = best_model.predict(X_test)

if PROBLEM_TYPE == 'classification':
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率 (Test Accuracy): {test_accuracy:.4f}")
    print("测试集混淆矩阵 (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
else:  # regression
    test_r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    print(f"测试集 R² 分数 (Test R2 Score): {test_r2:.4f}")
    print(f"测试集均方误差 (Test MSE): {test_mse:.4f}")

print("--- 任务结束 ---")
# 1. 导入所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---
# 2. 创建并可视化示例数据
# ---
# 设置随机种子以保证结果可复现
np.random.seed(42)

# 创建一个三维数据集，其中变量之间存在线性相关性
# 想象一下这是三个指标：'基础能力', '应用能力', '创新潜力'
# '应用能力' 高度依赖于 '基础能力'
# '创新潜力' 也部分依赖于 '基础能力'
mean = [0, 0, 0]
cov = [[1, 0.8, 0.5],
       [0.8, 1, 0.7],
       [0.5, 0.7, 1]]
X_orig = np.random.multivariate_normal(mean, cov, 100)

# 将数据转换为DataFrame，便于观察
df_orig = pd.DataFrame(X_orig, columns=['指标A', '指标B', '指标C'])
print("原始数据（前5行）:")
print(df_orig.head())
print("\n")

# 可视化原始三维数据
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_orig[:, 0], X_orig[:, 1], X_orig[:, 2])
ax.set_title('原始三维数据分布')
ax.set_xlabel('指标A')
ax.set_ylabel('指标B')
ax.set_zlabel('指标C')
plt.show()

# ---
# 3. 数据标准化
# ---
# PCA受变量尺度影响很大，因此标准化是必不可少的步骤
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_orig)

# ---
# 4. 执行PCA分析
# ---
# 初始化PCA。我们想将数据降至2维
# n_components参数指定了我们希望保留的主成分数量
n_components = 2
pca = PCA(n_components=n_components)

# 拟合模型到标准化后的数据上
pca.fit(X_scaled)

# 分析PCA的结果
# explained_variance_ratio_ 显示了每个主成分解释的方差百分比
print(f"每个主成分解释的方差比例: {pca.explained_variance_ratio_}")
print(f"总解释方差: {sum(pca.explained_variance_ratio_):.2f}")
print("\n")

# components_ 显示了主成分（特征向量），行代表主成分，列对应原始指标
# 每一行告诉我们这个主成分是由原始指标如何线性组合而成的
components_df = pd.DataFrame(pca.components_,
                             columns=df_orig.columns,
                             index=[f'主成分{i+1}' for i in range(n_components)])
print("主成分 (特征向量):")
print(components_df)
print("\n")

# ---
# 5. 数据转换与降维
# ---
# 使用训练好的PCA模型来转换（降维）数据
# 这就是计算“主成分得分”的步骤：Z_new = Z_scaled ⋅ V
X_pca = pca.transform(X_scaled)

# 将降维后的数据也放入DataFrame
df_pca = pd.DataFrame(X_pca, columns=[f'主成分{i+1}' for i in range(n_components)])
print("降维后的数据（主成分得分，前5行）:")
print(df_pca.head())
print("\n")


# ---
# 6. 可视化降维后的结果
# ---
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)
plt.title('PCA降维后的二维数据')
plt.xlabel('第一主成分 (PC1)')
plt.ylabel('第二主成分 (PC2)')
plt.grid(True)
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.show()


# (可选) 绘制碎石图(Scree Plot)来帮助选择主成分数量
# 为了绘制完整的碎石图，我们需要用所有可能的主成分来拟合一次PCA
pca_full = PCA()
pca_full.fit(X_scaled)
explained_variance_ratio_full = pca_full.explained_variance_ratio_

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio_full) + 1), explained_variance_ratio_full, marker='o', linestyle='--')
plt.title('碎石图 (Scree Plot)')
plt.xlabel('主成分数量')
plt.ylabel('解释的方差比例')
plt.xticks(range(1, len(explained_variance_ratio_full) + 1))
plt.grid(True)
plt.show()
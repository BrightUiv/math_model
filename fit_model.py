import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



# 一、定义函数模型: 难点，建立正确的函数模型
# 二、通过xlsx导入数据集
# 三、设置参数的初始值，方便之后的优化
# 四、通过curve_fit进行拟合

# 1. 定义你的复杂函数 (配方)
def damped_sine_func(x, a, b, c, d):
    """
    一个阻尼正弦函数模型
    a: 初始振幅
    b: 衰减系数
    c: 角频率
    d: 相位
    """
    return a * np.exp(-b * x) * np.sin(c * x + d)

# 2. 准备数据 (原材料)
# 生成一些带有噪声的模拟数据
x_data = np.linspace(0, 10, 100)
# 真实参数为 a=5, b=0.5, c=2.0, d=0.8
true_params = [5, 0.5, 2.0, 0.8]
y_true = damped_sine_func(x_data, *true_params)
# 加入一些随机噪声
noise = 0.3 * np.random.normal(size=x_data.size)
y_data = y_true + noise

# 3. 提供初始猜测值 (预热烤箱)
# 根据对图形的观察或先验知识来估计
initial_guess = [4, 0.6, 2.2, 0.7]

# 4. === 执行核心拟合命令 ===
popt, pcov = curve_fit(damped_sine_func, x_data, y_data, p0=initial_guess)


# 5. 分析结果 (品尝蛋糕和阅读质检报告)
print("找到的最优参数 (a, b, c, d):")
print(popt)
print("\n参数的协方差矩阵:")
print(pcov)

# 提取每个参数的标准差（不确定度）
param_std_dev = np.sqrt(np.diag(pcov))
print("\n每个参数的标准差:")
print(param_std_dev)

# 6. 可视化拟合效果
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Original Noisy Data', s=10) # 原始数据点
plt.plot(x_data, y_true, 'g--', label='True Curve (no noise)') # 真实的曲线
plt.plot(x_data, damped_sine_func(x_data, *popt), 'r-', label='Fitted Curve') # 拟合出的曲线
plt.title("Fit of a Damped Sine Wave")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
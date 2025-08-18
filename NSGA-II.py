# -----------------------------------------
# 1. 导入必要的库
# -----------------------------------------
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


# -----------------------------------------
# 2. 定义多目标优化问题
# -----------------------------------------
# 我们需要继承 pymoo 提供的 Problem 类来定义自己的问题。
# ElementwiseProblem 类使得我们每次只需处理一个解（而不是一批解），定义起来更简单。
class MyProblem(ElementwiseProblem):

    def __init__(self):
        # 调用父类的构造函数，并在这里定义问题的基本属性
        # n_var: 决策变量的数量 (我们有 x1 和 x2 两个变量)
        # n_obj: 优化目标的数量 (我们有 f1 和 f2 两个目标)
        # xl: 每个决策变量的下界 (lower bound)
        # xu: 每个决策变量的上界 (upper bound)
        #
        # !! 注意：这里我们为 xl 和 xu 提供了具体的值，
        # !! 这就解决了您之前遇到的 TypeError 问题。
        super().__init__(n_var=2,
                         n_obj=2,
                         xl=np.array([-5.0, -5.0]),  # 变量 x1, x2 的下界都是 -5
                         xu=np.array([5.0, 5.0]))  # 变量 x1, x2 的上界都是 +5

    def _evaluate(self, x, out, *args, **kwargs):
        # 这是核心函数，用于计算每个解 x 的目标函数值

        # x 是一个 NumPy 数组，包含了所有决策变量的值。
        # 例如: x = [x1, x2]

        # 目标 f1: 我们希望 f1 尽可能小
        # 这是一个凸函数，当 x1=1, x2=1 时取得最小值
        f1 = (x[0] - 1) ** 2 + (x[1] - 1) ** 2

        # 目标 f2: 我们也希望 f2 尽可能小
        # 这是一个凸函数，当 x1=-1, x2=-1 时取得最小值
        f2 = (x[0] + 1) ** 2 + (x[1] + 1) ** 2

        # 这两个目标是冲突的：
        # - 使 f1 最小的解 ([1, 1]) 会让 f2 的值变大。
        # - 使 f2 最小的解 ([-1, -1]) 会让 f1 的值变大。
        # 算法的目的就是找到这两个目标之间的“折中”解集。

        # 'out' 是一个字典，我们需要把计算出的目标函数值存入 'F' 键
        out["F"] = [f1, f2]


# -----------------------------------------
# 3. 主程序入口
# -----------------------------------------
if __name__ == '__main__':
    # 步骤 3.1: 实例化我们定义的问题
    problem = MyProblem()

    # 步骤 3.2: 实例化算法
    # 我们选择 NSGA-II 算法
    # pop_size: 种群大小，即每一代有多少个个体（解）。
    algorithm = NSGA2(pop_size=100)

    # 步骤 3.3: 定义终止条件
    # 这里我们设置为运行 200 代 (generations)
    termination = ("n_gen", 200)

    # 步骤 3.4: 开始执行优化
    # minimize 函数将问题、算法和终止条件串联起来
    # seed=1: 设置随机种子，保证每次运行结果可复现
    # verbose=True: 打印每一代的优化进程
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True)

    # -----------------------------------------
    # 4. 输出和可视化结果
    # -----------------------------------------
    print("\n优化完成!")

    # res.X 存储的是帕累托最优解集对应的决策变量值 (n_solutions, n_var)
    # res.F 存储的是帕累托最优解集对应的目标函数值 (n_solutions, n_obj)
    print("找到的帕累托最优解的数量:", len(res.F))
    # print("决策变量 (X): \n", res.X)
    # print("目标函数值 (F): \n", res.F)

    # 步骤 4.1: 可视化帕累托前沿 (Pareto Front)
    # 创建一个散点图对象
    plot = Scatter(title="Pareto Front: f1 vs f2",
                   xlabel="f1",
                   ylabel="f2")

    # 将找到的目标函数值添加到图表中
    plot.add(res.F)

    # 显示图表
    plot.show()
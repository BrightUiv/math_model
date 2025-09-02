import numpy as np


# todo: 替换为实际的适应度
def sphere_function(x):

    return np.sum(x ** 2)


# todo:替换为初始化函数
class Particle:
    """

    粒子类，用于存储粒子的所有信息。

    """
    def __init__(self, dim, bounds):
        """
        初始化一个粒子。

        参数:
        - dim (int): 问题的维度。
        - bounds (tuple): 一个元组 (min_bound, max_bound)，定义了搜索空间的边界。
        """
        min_bound, max_bound = bounds
        # 随机初始化粒子的位置
        self.position = np.random.uniform(low=min_bound, high=max_bound, size=dim)
        # 初始化粒子的速度
        self.velocity = np.random.uniform(low=-abs(max_bound - min_bound) * 0.1,
                                          high=abs(max_bound - min_bound) * 0.1,
                                          size=dim)
        # 计算初始位置的适应度值
        self.fitness = sphere_function(self.position)

        # 初始化个体的历史最优位置和适应度
        self.pbest_position = self.position.copy()
        self.pbest_fitness = self.fitness




def pso_optimizer(objective_func, dim, bounds, num_particles, max_iter, c1, c2, w_max, w_min):
    """
    使用线性递减惯性权重的粒子群算法进行优化。

    参数:
    - objective_func (function): 要优化的目标函数。
    - dim (int): 问题的维度。
    - bounds (tuple): 搜索空间的边界 (min_bound, max_bound)。
    - num_particles (int): 粒子数量（种群大小）。
    - max_iter (int): 最大迭代次数。
    - c1 (float): 认知系数（个体学习因子）。
    - c2 (float): 社会系数（群体学习因子）。
    - w_max (float): 惯性权重的最大值。
    - w_min (float): 惯性权重的最小值。

    返回:
    - gbest_position (np.array): 找到的全局最优位置。
    - gbest_fitness (float): 全局最优位置对应的适应度值。
    """
    min_bound, max_bound = bounds

    # --- 2. 初始化粒子群 ---
    swarm = [Particle(dim, bounds) for _ in range(num_particles)]

    # 初始化全局最优位置和适应度
    # 假设第一个粒子就是当前的全局最优
    gbest_position = swarm[0].pbest_position.copy()
    gbest_fitness = swarm[0].pbest_fitness

    # 寻找真正的初始全局最优
    for particle in swarm[1:]:
        if particle.pbest_fitness < gbest_fitness:
            gbest_fitness = particle.pbest_fitness
            gbest_position = particle.pbest_position.copy()

    # 用于记录每次迭代的最佳适应度，方便后续绘图分析
    convergence_curve = np.zeros(max_iter)

    # --- 3. 开始主循环 ---
    for i in range(max_iter):
        # a. 计算当前迭代的惯性权重 w (线性递减)
        w = w_max - (w_max - w_min) * (i / max_iter)

        for particle in swarm:
            # b. 更新粒子的速度
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            cognitive_velocity = c1 * r1 * (particle.pbest_position - particle.position)
            social_velocity = c2 * r2 * (gbest_position - particle.position)

            # 速度更新公式
            particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity

            # c. 更新粒子的位置
            particle.position = particle.position + particle.velocity

            # d. 边界处理：防止粒子飞出搜索空间
            particle.position = np.clip(particle.position, min_bound, max_bound)

            # e. 计算新位置的适应度
            current_fitness = objective_func(particle.position)

            # f. 更新个体最优 (pbest)
            if current_fitness < particle.pbest_fitness:
                particle.pbest_position = particle.position.copy()
                particle.pbest_fitness = current_fitness

        # g. 更新全局最优 (gbest)
        for particle in swarm:
            if particle.pbest_fitness < gbest_fitness:
                gbest_fitness = particle.pbest_fitness
                gbest_position = particle.pbest_position.copy()

        # 记录本次迭代的全局最优适应度
        convergence_curve[i] = gbest_fitness

        # (可选) 打印每次迭代的信息
        if (i + 1) % 10 == 0:
            print(f"迭代 {i + 1}/{max_iter} | 全局最优适应度: {gbest_fitness:.6f} | 惯性权重 w: {w:.4f}")

    return gbest_position, gbest_fitness, convergence_curve


# --- 4. 设置参数并运行算法 ---
if __name__ == "__main__":
    # 问题定义
    PROBLEM_DIMENSION = 10  # 问题维度
    SEARCH_BOUNDS = (-10, 10)  # 搜索空间的边界 [-10, 10]

    # PSO 算法参数
    NUM_PARTICLES = 50  # 粒子数量
    MAX_ITERATIONS = 200  # 最大迭代次数
    C1 = 2.0  # 认知系数
    C2 = 2.0  # 社会系数
    W_MAX = 0.9  # 惯性权重最大值
    W_MIN = 0.4  # 惯性权重最小值

    print("开始使用LDIW-PSO算法进行优化...")
    print("目标函数: Sphere Function")
    print(f"维度: {PROBLEM_DIMENSION}, 搜索范围: {SEARCH_BOUNDS}")
    print("-" * 30)

    # 运行PSO优化器
    best_position, best_fitness, curve = pso_optimizer(
        sphere_function,
        PROBLEM_DIMENSION,
        SEARCH_BOUNDS,
        NUM_PARTICLES,
        MAX_ITERATIONS,
        C1, C2, W_MAX, W_MIN
    )

    print("-" * 30)
    print("优化完成！")
    print(f"找到的最优位置 (gbest): \n{best_position}")
    print(f"找到的最优适应度值 (最小值): {best_fitness}")

    # (可选) 绘制收敛曲线
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(curve)
        plt.title("PSO Convergence Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Global Best Fitness")
        plt.grid(True)
        plt.show()
    except ImportError:
        print("\n提示: 如果您想查看收敛曲线图，请安装 matplotlib (`pip install matplotlib`)")
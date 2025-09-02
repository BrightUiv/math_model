import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# --- 1. 创建包含动态障碍物的自定义无人机环境 ---
class UAVGridEnvDynamic(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=10):
        super(UAVGridEnvDynamic, self).__init__()

        self.grid_size = grid_size
        self.window_size = 512

        # --- 核心创新点 1: 扩充状态空间 ---
        # 状态现在包含: [无人机x, 无人机y, 障碍物1_x, 障碍物1_y, 障碍物2_x, 障碍物2_y, ...]
        # 假设我们有2个动态障碍物
        self.num_dynamic_obstacles = 2
        observation_shape = (2 + self.num_dynamic_obstacles * 2,)
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=observation_shape, dtype=np.float32)

        # 动作空间保持不变
        self.action_space = spaces.Discrete(4)

        # 环境元素
        self.start_pos = np.array([0, 0])
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([grid_size - 1, grid_size - 1])

        self.static_obstacles = [np.array([3, 4]), np.array([4, 4]), np.array([5, 4]), np.array([6, 4])]

        # --- 核心创新点 2: 定义动态障碍物 ---
        self.dynamic_obstacles = []
        self._init_dynamic_obstacles()

        self.window = None
        self.clock = None

    def _init_dynamic_obstacles(self):
        """初始化/重置动态障碍物"""
        self.dynamic_obstacles = [
            # 障碍物1: 水平巡逻
            {'pos': np.array([2, 1]), 'direction': 1, 'path': [(2, i) for i in range(1, 8)]},
            # 障碍物2: 垂直巡逻
            {'pos': np.array([1, 6]), 'direction': 1, 'path': [(i, 6) for i in range(1, 8)]}
        ]

    def _move_obstacles(self):
        """在每一步更新动态障碍物的位置"""
        for obs in self.dynamic_obstacles:
            current_index = obs['path'].index(tuple(obs['pos']))
            if current_index + obs['direction'] >= len(obs['path']) or current_index + obs['direction'] < 0:
                obs['direction'] *= -1  # 碰到路径终点则反向

            new_index = current_index + obs['direction']
            obs['pos'] = np.array(obs['path'][new_index])

    def _get_obs(self):
        """构建包含所有动态信息的观测"""
        obs_list = [self.agent_pos]
        for obstacle in self.dynamic_obstacles:
            obs_list.append(obstacle['pos'])
        return np.concatenate(obs_list).astype(np.float32)

    def _get_info(self):
        return {"distance": np.linalg.norm(self.agent_pos - self.goal_pos)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self._init_dynamic_obstacles()  # 每次重置时，障碍物也回到初始位置
        return self._get_obs(), self._get_info()

    def step(self, action):
        # 1. 动态障碍物先移动
        self._move_obstacles()

        # 2. 无人机根据动作尝试移动
        proposed_pos = self.agent_pos.copy()
        if action == 0:
            proposed_pos[1] -= 1
        elif action == 1:
            proposed_pos[1] += 1
        elif action == 2:
            proposed_pos[0] -= 1
        elif action == 3:
            proposed_pos[0] += 1

        # 3. 检查碰撞和边界
        terminated = False
        reward = -0.01

        # 检查是否撞墙
        if not (0 <= proposed_pos[0] < self.grid_size and 0 <= proposed_pos[1] < self.grid_size):
            reward = -1.0
            terminated = True
        # 检查是否撞静态障碍物
        elif any(np.array_equal(proposed_pos, obs) for obs in self.static_obstacles):
            reward = -1.0
            terminated = True
        # 检查是否撞动态障碍物
        elif any(np.array_equal(proposed_pos, obs['pos']) for obs in self.dynamic_obstacles):
            reward = -1.0
            terminated = True
        # 未发生碰撞，更新位置
        else:
            self.agent_pos = proposed_pos

        # 检查是否到达终点
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 1.0
            terminated = True

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.grid_size

        pygame.draw.rect(canvas, (0, 255, 0),
                         pygame.Rect(self.goal_pos[0] * pix_square_size, self.goal_pos[1] * pix_square_size,
                                     pix_square_size, pix_square_size))
        for obs in self.static_obstacles:
            pygame.draw.rect(canvas, (0, 0, 0),
                             pygame.Rect(obs[0] * pix_square_size, obs[1] * pix_square_size, pix_square_size,
                                         pix_square_size))
        # 用红色表示动态障碍物
        for obs in self.dynamic_obstacles:
            pygame.draw.rect(canvas, (255, 0, 0),
                             pygame.Rect(obs['pos'][0] * pix_square_size, obs['pos'][1] * pix_square_size,
                                         pix_square_size, pix_square_size))

        pygame.draw.circle(canvas, (0, 0, 255),
                           ((self.agent_pos[0] + 0.5) * pix_square_size, (self.agent_pos[1] + 0.5) * pix_square_size),
                           pix_square_size / 3)

        for x in range(self.grid_size + 1):
            pygame.draw.line(canvas, (128, 128, 128), (0, x * pix_square_size), (self.window_size, x * pix_square_size),
                             width=1)
            pygame.draw.line(canvas, (128, 128, 128), (x * pix_square_size, 0), (x * pix_square_size, self.window_size),
                             width=1)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# --- 2. 主执行函数 ---
if __name__ == "__main__":
    env = UAVGridEnvDynamic()
    check_env(env)

    # --- 核心创新点 3: 使用PPO算法 ---
    # PPO的"MlpPolicy"是一个Actor-Critic网络结构，下面会详细讲解
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./uav_ppo_tensorboard/")

    # 动态环境更复杂，需要更多的训练步数
    print("--- 开始训练PPO模型 (这可能需要几分钟) ---")
    model.learn(total_timesteps=100000, progress_bar=True)
    print("--- 训练完成 ---")

    model.save("ppo_uav_dynamic_model")
    print("模型已保存为 ppo_uav_dynamic_model.zip")

    print("\n--- 开始测试训练好的PPO模型 ---")
    obs, info = env.reset()
    for i in range(500):
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"回合在 {i + 1} 步后结束. Reward: {reward}")
            time.sleep(2)
            obs, info = env.reset()

    env.close()
import numpy as np
import random

class CliffWalkingEnv:
    """悬崖漫步环境构建"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.start_pos = (self.nrow - 1, 0)
        self.goal_pos = (self.nrow - 1, self.ncol - 1)
        self.reset()

    def reset(self):
        """重置环境到初始状态"""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        """执行动作：0-上, 1-下, 2-左, 3-右"""
        row, col = self.agent_pos
        if action == 0:  # 上
            row = max(row - 1, 0)
        elif action == 1:  # 下
            row = min(row + 1, self.nrow - 1)
        elif action == 2:  # 左
            col = max(col - 1, 0)
        elif action == 3:  # 右
            col = min(col + 1, self.ncol - 1)
        
        self.agent_pos = (row, col)
        
        # 掉入悬崖
        if self.agent_pos[0] == self.nrow - 1 and 0 < self.agent_pos[1] < self.ncol - 1:
            return self.start_pos, -100, True 
        # 到达终点
        elif self.agent_pos == self.goal_pos:
            return self.goal_pos, 0, True 
        # 普通移动
        else:
            return self.agent_pos, -1, False

class QLearningAgent:
    """Q-learning 智能体"""
    def __init__(self, ncol, nrow, actions, lr=0.1, gamma=0.9, epsilon=0.1):
        self.ncol = ncol
        self.nrow = nrow
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((nrow, ncol, len(actions)))

    def choose_action(self, state):
        row, col = state
        if random.random() < self.epsilon:
            action = random.randint(0, len(self.actions) - 1)
        else:
            action = np.argmax(self.Q[row, col, :])
        return action

    def update(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        max_next_q = np.max(self.Q[next_row, next_col, :])
        current_q = self.Q[row, col, action]
        self.Q[row, col, action] = current_q + self.lr * (
            reward + self.gamma * max_next_q - current_q
        )

# --- 训练阶段 ---
env = CliffWalkingEnv()
agent = QLearningAgent(ncol=12, nrow=4, actions=[0, 1, 2, 3])
episodes = 400  # 增加训练次数让结果更收敛

for episode in range(episodes):
    state = env.reset()
    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        if done:
            break

# --- 🗺️ 修复后的可视化打印函数 ---
def print_grid_policy(agent):
    actions_map = {0: '^', 1: 'v', 2: '<', 3: '>'}
    print("\n[ 4x12 网格最优策略图 ]")
    print("说明: S=起点, G=终点, C=悬崖, 箭头代表智能体在该格子倾向的移动方向\n")
    
    for r in range(4):
        row_str = ""
        for c in range(12):
            if r == 3 and c == 0:
                row_str += " S  "  # 起点
            elif r == 3 and c == 11:
                row_str += " G  "  # 终点
            elif r == 3 and 0 < c < 11:
                row_str += " C  "  # 悬崖
            else:
                # 获取该格子的最优动作
                best_action = np.argmax(agent.Q[r, c, :])
                row_str += f" {actions_map[best_action]}  "
        print(row_str)

print_grid_policy(agent)

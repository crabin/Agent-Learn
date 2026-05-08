"""
mdp_rl_demo_clean.py

本文件用于演示强化学习中的核心数学概念和运行逻辑。

包含内容：

1. MDP：马尔可夫决策过程
2. Policy：策略 pi(a|s)
3. Trajectory：轨迹
4. Return：回报 G
5. Value Function：状态价值函数 V(s)
6. Action-Value Function：状态-动作价值函数 Q(s,a)
7. Advantage Function：优势函数 A(s,a)
8. Bellman Expectation Equation：Bellman 期望方程
9. Bellman Optimality Equation：Bellman 最优方程
10. Value Iteration：值迭代
11. Q-learning：无模型强化学习方法

设计目标：

- 代码注释尽量详细，适合作为学习笔记。
- 运行输出不要过多刷屏。
- 控制台输出重点展示：
    1. 程序正在执行什么阶段
    2. 每个阶段的变化趋势
    3. 最终得到的价值函数和策略
"""

import random
from collections import defaultdict


# ============================================================
# 1. 定义 GridWorld MDP 环境
# ============================================================

class GridWorldMDP:
    """
    一个简单的 GridWorld 环境。

    状态 S：
        智能体当前所在的网格坐标，例如 (0, 0)。

    动作 A：
        up, down, left, right。

    转移概率 P：
        这里使用确定性转移。
        也就是说，给定状态 s 和动作 a，下一状态 s' 是确定的。

    奖励函数 R：
        每走一步奖励 -1；
        到达终点奖励 +10。

    折扣因子 gamma：
        控制智能体对未来奖励的重视程度。
    """

    def __init__(self, rows=4, cols=4, start=(0, 0), goal=(3, 3), gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.gamma = gamma

        self.actions = ["up", "down", "left", "right"]

        self.states = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
        ]

    def is_terminal(self, state):
        """
        判断某个状态是否为终止状态。
        """
        return state == self.goal

    def step(self, state, action):
        """
        环境一步转移。

        输入：
            state: 当前状态 s
            action: 当前动作 a

        输出：
            next_state: 下一状态 s'
            reward: 当前动作获得的奖励 r
            done: 是否到达终止状态

        对应 MDP 里的：
            P(s' | s, a)
            R(s, a, s')
        """

        if self.is_terminal(state):
            return state, 0, True

        r, c = state

        if action == "up":
            nr, nc = r - 1, c
        elif action == "down":
            nr, nc = r + 1, c
        elif action == "left":
            nr, nc = r, c - 1
        elif action == "right":
            nr, nc = r, c + 1
        else:
            raise ValueError(f"Unknown action: {action}")

        # 越界则原地不动
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            next_state = state
        else:
            next_state = (nr, nc)

        if next_state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return next_state, reward, done

    def transition_prob(self, state, action, next_state):
        """
        转移概率 P(s' | s, a)。

        当前环境是确定性的：
            如果执行 action 后确实到达 next_state，概率为 1；
            否则为 0。
        """

        actual_next_state, _, _ = self.step(state, action)
        return 1.0 if actual_next_state == next_state else 0.0

    def reward_function(self, state, action, next_state):
        """
        奖励函数 R(s, a, s')。
        """

        _, reward, _ = self.step(state, action)
        return reward


# ============================================================
# 2. 策略定义
# ============================================================

class RandomPolicy:
    """
    随机策略。

    在每个状态下，所有动作被选择的概率相同：

        pi(a|s) = 1 / |A|
    """

    def __init__(self, actions):
        self.actions = actions

    def action_prob(self, state, action):
        """
        返回 pi(a|s)。
        """
        return 1.0 / len(self.actions)

    def sample_action(self, state):
        """
        根据随机策略采样动作。
        """
        return random.choice(self.actions)


# ============================================================
# 3. 轨迹与回报
# ============================================================

def generate_trajectory(env, policy, max_steps=50):
    """
    生成一条轨迹。

    轨迹形式：

        tau = (s0, a0, r0, s1, a1, r1, ...)

    它表示智能体从起点开始，按照策略 pi 与环境交互的完整过程。
    """

    trajectory = []
    state = env.start

    for _ in range(max_steps):
        action = policy.sample_action(state)
        next_state, reward, done = env.step(state, action)

        trajectory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        })

        state = next_state

        if done:
            break

    return trajectory


def compute_return(trajectory, gamma):
    """
    计算轨迹的折扣回报：

        G = r0 + gamma * r1 + gamma^2 * r2 + ...

    注意：
        这里的 r0 表示 trajectory[0]["reward"]。
    """

    G = 0.0

    for t, step in enumerate(trajectory):
        G += (gamma ** t) * step["reward"]

    return G


def summarize_trajectory(trajectory):
    """
    用简洁方式总结轨迹。

    不打印每一步细节，只打印：
        - 总步数
        - 是否到达终点
        - 起点和终点
    """

    if not trajectory:
        return "空轨迹"

    start = trajectory[0]["state"]
    end = trajectory[-1]["next_state"]
    done = trajectory[-1]["done"]

    return {
        "steps": len(trajectory),
        "start": start,
        "end": end,
        "reached_goal": done,
    }


# ============================================================
# 4. Bellman 期望方程：策略评估
# ============================================================

def policy_evaluation(env, policy, theta=1e-6, max_iterations=1000, log_every=20):
    """
    使用 Bellman 期望方程计算给定策略下的状态价值函数 V^pi(s)。

    Bellman 期望方程：

        V^pi(s)
        = sum_a pi(a|s) sum_s' P(s'|s,a)
          [R(s,a,s') + gamma * V^pi(s')]

    这个函数的运行逻辑：

        1. 初始化所有 V(s) = 0
        2. 对每个状态使用 Bellman 期望方程更新 V(s)
        3. 重复更新，直到变化量 delta 足够小
        4. 返回稳定后的 V(s)

    控制台只输出关键变化：
        - 第几轮迭代
        - 最大价值变化 delta
        - 起点状态 V(start) 的变化
    """

    V = {state: 0.0 for state in env.states}

    print("\n[阶段 1] Policy Evaluation：用 Bellman 期望方程评估随机策略")
    print("目标：计算在随机策略下，每个状态的长期价值 V^pi(s)。")

    for iteration in range(max_iterations):
        delta = 0.0

        for state in env.states:
            if env.is_terminal(state):
                continue

            old_value = V[state]
            new_value = 0.0

            for action in env.actions:
                action_prob = policy.action_prob(state, action)

                for next_state in env.states:
                    prob = env.transition_prob(state, action, next_state)
                    reward = env.reward_function(state, action, next_state)

                    new_value += action_prob * prob * (
                        reward + env.gamma * V[next_state]
                    )

            V[state] = new_value
            delta = max(delta, abs(old_value - new_value))

        if iteration % log_every == 0:
            print(
                f"  iter={iteration:03d} | "
                f"max_delta={delta:.6f} | "
                f"V(start)={V[env.start]:.4f}"
            )

        if delta < theta:
            print(
                f"  收敛完成：iter={iteration}, "
                f"max_delta={delta:.8f}, "
                f"V(start)={V[env.start]:.4f}"
            )
            break

    return V


# ============================================================
# 5. 根据 V(s) 计算 Q(s,a)
# ============================================================

def compute_q_from_v(env, V):
    """
    根据状态价值函数 V(s)，计算状态-动作价值函数 Q(s,a)。

    公式：

        Q^pi(s,a)
        = sum_s' P(s'|s,a)
          [R(s,a,s') + gamma * V^pi(s')]

    含义：

        V(s) 只回答：
            当前状态整体好不好？

        Q(s,a) 回答：
            在当前状态下，具体做某个动作好不好？
    """

    Q = {}

    for state in env.states:
        for action in env.actions:
            q_value = 0.0

            for next_state in env.states:
                prob = env.transition_prob(state, action, next_state)
                reward = env.reward_function(state, action, next_state)

                q_value += prob * (
                    reward + env.gamma * V[next_state]
                )

            Q[(state, action)] = q_value

    return Q


# ============================================================
# 6. 优势函数 A(s,a)
# ============================================================

def compute_advantage(env, V, Q):
    """
    计算优势函数：

        A^pi(s,a) = Q^pi(s,a) - V^pi(s)

    含义：

        A(s,a) > 0：
            说明动作 a 比当前策略平均表现更好。

        A(s,a) < 0：
            说明动作 a 比当前策略平均表现更差。

    很多策略梯度算法，例如 Actor-Critic、A2C、PPO，
    都会使用优势函数来降低梯度估计方差。
    """

    A = {}

    for state in env.states:
        for action in env.actions:
            A[(state, action)] = Q[(state, action)] - V[state]

    return A


# ============================================================
# 7. Bellman 最优方程：值迭代
# ============================================================

def value_iteration(env, theta=1e-6, max_iterations=1000, log_every=10):
    """
    使用 Bellman 最优方程计算最优价值函数 V*(s)。

    Bellman 最优方程：

        V*(s) = max_a sum_s' P(s'|s,a)
                [R(s,a,s') + gamma * V*(s')]

    和 Policy Evaluation 的区别：

        Policy Evaluation：
            评估一个固定策略 pi，动作按 pi(a|s) 加权平均。

        Value Iteration：
            不再固定策略，而是在每个状态直接选择价值最大的动作。

    最终可以由 V*(s) 导出最优策略：

        pi*(s) = argmax_a Q*(s,a)
    """

    V = {state: 0.0 for state in env.states}

    print("\n[阶段 2] Value Iteration：用 Bellman 最优方程寻找最优策略")
    print("目标：不断选择更优动作，使 V(s) 逼近最优价值 V*(s)。")

    for iteration in range(max_iterations):
        delta = 0.0

        for state in env.states:
            if env.is_terminal(state):
                continue

            old_value = V[state]

            action_values = []

            for action in env.actions:
                q_value = 0.0

                for next_state in env.states:
                    prob = env.transition_prob(state, action, next_state)
                    reward = env.reward_function(state, action, next_state)

                    q_value += prob * (
                        reward + env.gamma * V[next_state]
                    )

                action_values.append(q_value)

            V[state] = max(action_values)
            delta = max(delta, abs(old_value - V[state]))

        if iteration % log_every == 0:
            print(
                f"  iter={iteration:03d} | "
                f"max_delta={delta:.6f} | "
                f"V*(start)={V[env.start]:.4f}"
            )

        if delta < theta:
            print(
                f"  收敛完成：iter={iteration}, "
                f"max_delta={delta:.8f}, "
                f"V*(start)={V[env.start]:.4f}"
            )
            break

    optimal_policy = derive_greedy_policy_from_v(env, V)

    return V, optimal_policy


def derive_greedy_policy_from_v(env, V):
    """
    根据 V(s) 导出贪心策略。

    对每个状态 s：

        pi(s) = argmax_a sum_s' P(s'|s,a)
                [R(s,a,s') + gamma * V(s')]
    """

    policy = {}

    for state in env.states:
        if env.is_terminal(state):
            policy[state] = "terminal"
            continue

        best_action = None
        best_value = float("-inf")

        for action in env.actions:
            q_value = 0.0

            for next_state in env.states:
                prob = env.transition_prob(state, action, next_state)
                reward = env.reward_function(state, action, next_state)

                q_value += prob * (
                    reward + env.gamma * V[next_state]
                )

            if q_value > best_value:
                best_value = q_value
                best_action = action

        policy[state] = best_action

    return policy


# ============================================================
# 8. Q-learning
# ============================================================

def q_learning(
    env,
    episodes=500,
    alpha=0.1,
    epsilon=0.1,
    max_steps=50,
    log_every=100
):
    """
    Q-learning 是一种无模型强化学习算法。

    它和 Value Iteration 的重要区别：

        Value Iteration：
            需要知道环境模型 P(s'|s,a)。

        Q-learning：
            不需要提前知道 P(s'|s,a)，只需要不断和环境交互，
            根据实际经验更新 Q(s,a)。

    Q-learning 更新公式：

        Q(s,a) <- Q(s,a) + alpha *
                  [r + gamma * max_a' Q(s',a') - Q(s,a)]

    其中：

        alpha：
            学习率。

        gamma：
            折扣因子。

        epsilon：
            探索率。

    epsilon-greedy 策略：

        以 epsilon 的概率随机探索；
        以 1 - epsilon 的概率选择当前 Q 值最大的动作。
    """

    Q = defaultdict(float)

    print("\n[阶段 3] Q-learning：不使用转移概率，通过交互学习 Q(s,a)")
    print("目标：通过试错，让起点到终点的路径越来越稳定。")

    for episode in range(1, episodes + 1):
        state = env.start
        total_reward = 0
        steps_taken = 0
        reached_goal = False

        for _ in range(max_steps):

            # epsilon-greedy 选择动作
            if random.random() < epsilon:
                action = random.choice(env.actions)
            else:
                action = max(
                    env.actions,
                    key=lambda a: Q[(state, a)]
                )

            next_state, reward, done = env.step(state, action)

            best_next_q = max(
                Q[(next_state, next_action)]
                for next_action in env.actions
            )

            td_target = reward + env.gamma * best_next_q
            td_error = td_target - Q[(state, action)]

            Q[(state, action)] += alpha * td_error

            total_reward += reward
            steps_taken += 1
            state = next_state

            if done:
                reached_goal = True
                break

        if episode % log_every == 0:
            start_best_action = max(
                env.actions,
                key=lambda a: Q[(env.start, a)]
            )

            start_best_q = Q[(env.start, start_best_action)]

            print(
                f"  episode={episode:04d} | "
                f"reward={total_reward:4d} | "
                f"steps={steps_taken:2d} | "
                f"goal={str(reached_goal):5s} | "
                f"best_start_action={start_best_action:5s} | "
                f"Q(start,best)={start_best_q:.4f}"
            )

    learned_policy = derive_greedy_policy_from_q(env, Q)

    return Q, learned_policy


def derive_greedy_policy_from_q(env, Q):
    """
    根据 Q(s,a) 导出贪心策略：

        pi(s) = argmax_a Q(s,a)
    """

    policy = {}

    for state in env.states:
        if env.is_terminal(state):
            policy[state] = "terminal"
        else:
            policy[state] = max(
                env.actions,
                key=lambda a: Q[(state, a)]
            )

    return policy


# ============================================================
# 9. 打印工具函数
# ============================================================

def print_value_table(env, V, title):
    """
    用网格形式打印价值函数。

    每个格子的数字表示该状态的长期价值。
    越接近目标，通常价值越高。
    """

    print(f"\n{title}")
    print("-" * 55)

    for r in range(env.rows):
        row_values = []

        for c in range(env.cols):
            state = (r, c)

            if state == env.goal:
                row_values.append("   G   ")
            else:
                row_values.append(f"{V[state]:7.2f}")

        print(" | ".join(row_values))


def print_policy_table(env, policy, title):
    """
    用网格形式打印策略。

    箭头含义：
        ↑：向上
        ↓：向下
        ←：向左
        →：向右
        G：终点
    """

    action_symbol = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→",
        "terminal": "G",
    }

    print(f"\n{title}")
    print("-" * 55)

    for r in range(env.rows):
        row_values = []

        for c in range(env.cols):
            state = (r, c)
            action = policy[state]
            row_values.append(f"   {action_symbol[action]}   ")

        print(" | ".join(row_values))


def print_action_analysis(env, V, Q, A, state):
    """
    打印某一个状态下的动作分析。

    包括：

        Q(s,a)：做这个动作的长期收益
        A(s,a)：这个动作相比平均策略的优势
    """

    print(f"\n状态 {state} 下的动作分析")
    print("-" * 55)
    print(f"V({state}) = {V[state]:.4f}")
    print()

    for action in env.actions:
        print(
            f"action={action:5s} | "
            f"Q={Q[(state, action)]:8.4f} | "
            f"A={A[(state, action)]:8.4f}"
        )


def print_shortest_path(env, policy, title, max_steps=30):
    """
    根据策略从起点走一遍，展示最终路径。

    这个函数用于直观展示：
        策略到底让智能体怎么走？
    """

    state = env.start
    path = [state]

    for _ in range(max_steps):
        if env.is_terminal(state):
            break

        action = policy[state]
        next_state, _, done = env.step(state, action)
        path.append(next_state)
        state = next_state

        if done:
            break

    print(f"\n{title}")
    print("-" * 55)
    print(" -> ".join(str(p) for p in path))

    if path[-1] == env.goal:
        print(f"结果：到达终点，总步数 = {len(path) - 1}")
    else:
        print("结果：没有在限定步数内到达终点")


# ============================================================
# 10. 主函数
# ============================================================

def main():
    """
    主运行逻辑。

    相比原始版本，本版本不会打印大量中间细节。
    控制台输出更关注：

        1. 当前算法阶段
        2. 每个阶段的收敛变化
        3. 最终价值函数
        4. 最终策略
        5. 从起点到终点的实际路径
    """

    random.seed(42)

    env = GridWorldMDP(
        rows=4,
        cols=4,
        start=(0, 0),
        goal=(3, 3),
        gamma=0.9
    )

    print("=" * 80)
    print("强化学习 MDP / Bellman / Q-learning 示例")
    print("=" * 80)
    print("环境：4x4 GridWorld")
    print("起点：(0, 0)")
    print("终点：(3, 3)")
    print("奖励：普通移动 -1，到达终点 +10")
    print("折扣因子 gamma = 0.9")

    # --------------------------------------------------------
    # 1. 生成一条随机策略轨迹
    # --------------------------------------------------------

    random_policy = RandomPolicy(env.actions)

    trajectory = generate_trajectory(
        env=env,
        policy=random_policy,
        max_steps=30
    )

    G = compute_return(trajectory, env.gamma)
    trajectory_summary = summarize_trajectory(trajectory)

    print("\n[预热] 使用随机策略生成一条轨迹")
    print("-" * 55)
    print(f"轨迹摘要：{trajectory_summary}")
    print(f"该轨迹的折扣回报 G = {G:.4f}")
    print("含义：随机走通常不稳定，可能很久到不了终点，因此回报可能较低。")

    # --------------------------------------------------------
    # 2. 策略评估：计算随机策略下的 V(s)
    # --------------------------------------------------------

    V_random = policy_evaluation(
        env=env,
        policy=random_policy,
        theta=1e-6,
        max_iterations=1000,
        log_every=20
    )

    print_value_table(
        env,
        V_random,
        title="随机策略下的状态价值 V^pi(s)"
    )

    # --------------------------------------------------------
    # 3. 根据 V(s) 计算 Q(s,a) 和 A(s,a)
    # --------------------------------------------------------

    print("\n[阶段 1 补充] 由 V(s) 推出 Q(s,a) 和优势函数 A(s,a)")
    print("目标：分析在起点状态下，不同动作的好坏。")

    Q_random = compute_q_from_v(env, V_random)
    A_random = compute_advantage(env, V_random, Q_random)

    print_action_analysis(
        env,
        V_random,
        Q_random,
        A_random,
        state=env.start
    )

    # --------------------------------------------------------
    # 4. 值迭代：求最优价值函数和最优策略
    # --------------------------------------------------------

    V_star, optimal_policy = value_iteration(
        env=env,
        theta=1e-6,
        max_iterations=1000,
        log_every=5
    )

    print_value_table(
        env,
        V_star,
        title="最优状态价值 V*(s)"
    )

    print_policy_table(
        env,
        optimal_policy,
        title="由 Bellman 最优方程得到的最优策略 pi*(s)"
    )

    print_shortest_path(
        env,
        optimal_policy,
        title="按照 Value Iteration 得到的策略，从起点实际走一遍"
    )

    # --------------------------------------------------------
    # 5. Q-learning：通过交互学习策略
    # --------------------------------------------------------

    Q_learned, learned_policy = q_learning(
        env=env,
        episodes=1000,
        alpha=0.1,
        epsilon=0.1,
        max_steps=50,
        log_every=100
    )

    print_policy_table(
        env,
        learned_policy,
        title="Q-learning 学到的策略"
    )

    print_shortest_path(
        env,
        learned_policy,
        title="按照 Q-learning 学到的策略，从起点实际走一遍"
    )

    # --------------------------------------------------------
    # 6. 最终总结
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("运行逻辑总结")
    print("=" * 80)
    print("""
1. 随机策略：
   智能体随便走，因此轨迹不稳定，回报通常较低。

2. Policy Evaluation：
   固定随机策略，用 Bellman 期望方程计算每个状态的价值 V^pi(s)。
   这个过程回答：如果继续按随机策略走，当前状态长期来看值多少钱？

3. Q(s,a) 和 A(s,a)：
   Q(s,a) 进一步回答：在当前状态下，具体做某个动作值多少钱？
   A(s,a) 回答：这个动作比当前策略平均水平好多少？

4. Value Iteration：
   使用 Bellman 最优方程，不再平均所有动作，而是每次选最好的动作。
   最终得到 V*(s) 和最优策略 pi*(s)。

5. Q-learning：
   不需要知道转移概率 P(s'|s,a)，只通过真实交互更新 Q(s,a)。
   学到 Q(s,a) 后，再用 argmax_a Q(s,a) 得到策略。

一句话：
   强化学习本质上是在学习“状态下该选什么动作”，
   也就是学习一个能最大化长期回报的策略。
""")


if __name__ == "__main__":
    main()
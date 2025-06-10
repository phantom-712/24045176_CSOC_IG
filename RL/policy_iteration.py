import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import pygame
import time

# -------------------------------
# Base Custom FrozenLake Class
# -------------------------------
class CustomFrozenLake(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.map = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG"
        ]
        self.setup_environment()

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 512
        self.cell_size = self.window_size // self.nrow

    def setup_environment(self):
        self.nrow = len(self.map)
        self.ncol = len(self.map[0])
        self.nS = self.nrow * self.ncol
        self.nA = 4
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        self.P = self._build_transition_model()
        self.state = self._to_state(0, 0)

    def _to_state(self, r, c):
        return r * self.ncol + c

    def _to_pos(self, s):
        return divmod(s, self.ncol)

    def _build_transition_model(self):
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        slip_probs = [0.1, 0.8, 0.1]
        directions = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

        for r in range(self.nrow):
            for c in range(self.ncol):
                s = self._to_state(r, c)
                cell = self.map[r][c]
                if cell in "GH":
                    for a in range(self.nA):
                        P[s][a] = [(1.0, s, 0, True)]
                    continue

                for a in range(self.nA):
                    transitions = []
                    for i, offset in enumerate([-1, 0, 1]):
                        a_eff = (a + offset) % 4
                        dr, dc = directions[a_eff]
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.nrow and 0 <= nc < self.ncol:
                            ns = self._to_state(nr, nc)
                            next_cell = self.map[nr][nc]
                        else:
                            ns = s
                            next_cell = cell

                        reward = 1.0 if next_cell == 'G' else 0.0
                        done = next_cell in "GH"
                        transitions.append((slip_probs[i], ns, reward, done))
                    P[s][a] = transitions
        return P

    def reset(self, seed=None, options=None):
        self.state = self._to_state(0, 0)
        return self.state, {}

    def step(self, action):
        transitions = self.P[self.state][action]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        _, s_prime, reward, done = transitions[i]
        self.state = s_prime
        return s_prime, reward, done, False, {}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        colors = {'S': (0, 255, 0), 'F': (180, 180, 255), 'H': (0, 0, 0), 'G': (255, 215, 0)}
        self.window.fill((255, 255, 255))
        for r in range(self.nrow):
            for c in range(self.ncol):
                cell = self.map[r][c]
                color = colors.get(cell, (200, 200, 200))
                pygame.draw.rect(self.window, color, pygame.Rect(
                    c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

        r, c = self._to_pos(self.state)
        pygame.draw.circle(self.window, (255, 0, 0), (
            c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2), self.cell_size // 3)

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()

# ------------------------------------
# Expanded 10x10 FrozenLake Version
# ------------------------------------
class CustomFrozenLakeExpanded(CustomFrozenLake):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.map = [
            "SFFFFFFFFF",
            "FFFFFFFFFF",
            "FFFHFFFFFF",
            "FFFFFHFFFF",
            "FFFHFFFFFH",
            "FHHFFFFFHF",
            "FHFFFHFHFF",
            "FFFFHFFFFF",
            "FFFFFHFFFF",
            "FFFFFFFFFG"
        ]
        self.setup_environment()
        self.cell_size = self.window_size // self.nrow


# -----------------------
# Register both envs
# -----------------------
register(id='CustomFrozenLake8x8-v0', entry_point=__name__ + ':CustomFrozenLake', kwargs={'render_mode': None})
register(id='CustomFrozenLake10x10-v0', entry_point=__name__ + ':CustomFrozenLakeExpanded', kwargs={'render_mode': None})


# -----------------------
# Policy Iteration (DP)
# -----------------------

def timed_policy_iteration(env, gamma=0.99, theta=1e-8):
    nS, nA = env.observation_space.n, env.action_space.n
    policy = np.zeros(nS, dtype=int)
    V = np.zeros(nS)
    iterations = 0
    start_time = time.time()

    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(nS):
                a = policy[s]
                v = V[s]
                V[s] = sum(p * (r + gamma * V[s_]) for p, s_, r, d in env.P[s][a])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(nS):
            old_action = policy[s]
            action_values = np.array([
                sum(p * (r + gamma * V[s_]) for p, s_, r, d in env.P[s][a])
                for a in range(nA)
            ])
            best_action = np.argmax(action_values)
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        iterations += 1
        if policy_stable:
            break

    duration = time.time() - start_time
    return policy, V, iterations, duration

def evaluate_policy(env, policy, episodes=100):
    total_rewards = []
    steps = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = policy[obs]
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1

        total_rewards.append(total_reward)
        steps.append(step_count)

    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(steps)
    success_rate = np.mean(total_rewards) * 100

    return {
        "Average Reward": avg_reward,
        "Average Episode Length": avg_length,
        "Success Rate (%)": success_rate
    }


# ----------------------------
# Visual Playback of Policy
# ----------------------------
def run_visual_policy(env, policy, delay=0.5):
    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action = policy[obs]
        obs, reward, done, _, _ = env.step(action)
        time.sleep(delay)
    env.render()
    time.sleep(2)
    env.close()


# ----------------------------
# Run both environments
# ----------------------------
if __name__ == "__main__":
    print("8x8 Frozen Lake")
    env8 = CustomFrozenLake(render_mode="human")
    policy8, V8, iter8, dur8 = timed_policy_iteration(env8)
    print(f"Policy Iteration on 8x8: {iter8} improvement steps, {dur8:.3f} seconds")
    eval_metrics8 = evaluate_policy(env8, policy8)
    print("Evaluation Metrics (8x8):")
    for k, v in eval_metrics8.items():
        print(f"{k}: {v}")
    run_visual_policy(env8, policy8)

    print("\n10x10 Frozen Lake")
    env10 = CustomFrozenLakeExpanded(render_mode="human")
    policy10, V10, iter10, dur10 = timed_policy_iteration(env10)
    print(f"Policy Iteration on 10x10: {iter10} improvement steps, {dur10:.3f} seconds")
    eval_metrics10 = evaluate_policy(env10, policy10)
    print("Evaluation Metrics (10x10):")
    for k, v in eval_metrics10.items():
        print(f"{k}: {v}")
    run_visual_policy(env10, policy10)

import numpy as np
import gymnasium as gym

def policy_iteration(env, gamma=0.99, theta=1e-8):
    nS = env.observation_space.n
    nA = env.action_space.n
    policy = np.zeros(nS, dtype=int)
    V = np.zeros(nS)
    policy_eval_iterations = 0
    policy_improvement_steps = 0

    while True:
        # Policy Evaluation
        while True:
            delta = 0
            policy_eval_iterations += 1
            for s in range(nS):
                a = policy[s]
                v = sum(p * (r + gamma * V[s_]) for p, s_, r, d in env.P[s][a])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        policy_improvement_steps += 1
        for s in range(nS):
            old_action = policy[s]
            action_values = np.zeros(nA)
            for a in range(nA):
                action_values[a] = sum(p * (r + gamma * V[s_]) for p, s_, r, d in env.P[s][a])
            best_action = np.argmax(action_values)
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return policy, V, policy_eval_iterations, policy_improvement_steps

def run_policy(env, policy, episodes=2, render=True):
    episode_lengths = []
    rewards = []
    successes = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        if render:
            print(f"\nEpisode {ep+1}")
            env.render()
        while not done:
            action = policy[obs]
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            if render:
                env.render()
        episode_lengths.append(steps)
        rewards.append(total_reward)
        if total_reward > 0:
            successes += 1

    avg_length = np.mean(episode_lengths)
    success_rate = successes / episodes
    avg_reward = np.mean(rewards)
    var_length = np.var(episode_lengths)

    print("\n--- Metrics ---")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Reward per Episode: {avg_reward:.2f}")
    print(f"Variance in Episode Length: {var_length:.2f}")

    return avg_length, success_rate, avg_reward, var_length

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    env = env.unwrapped

    policy, V, eval_iters, improve_steps = policy_iteration(env)
    print("Policy Evaluation Iterations:", eval_iters)
    print("Policy Improvement Steps:", improve_steps)
    print("Optimal Policy:")
    print(policy.reshape((4, 4)))

    run_policy(env, policy, episodes=10, render=True)
    env.close()

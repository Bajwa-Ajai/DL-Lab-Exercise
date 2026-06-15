import os
from datetime import datetime
import gymnasium as gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import MLP
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v1", render_mode="human")

    state_dim = 4
    num_actions = 2

    # TODO: load DQN agent
    # ...
    Q = MLP(state_dim=state_dim, action_dim=num_actions)
    Q_target = MLP(state_dim=state_dim, action_dim=num_actions)
    agent = DQNAgent(Q=Q, Q_target=Q_target, num_actions=num_actions)
    agent.load("models/dqn_agent_cartpole.pt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env, agent, deterministic=True, do_training=False, rendering=True
        )
        episode_rewards.append(stats.episode_reward)
        print(f"Episode {i+1}: reward = {stats.episode_reward:.2f}")

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    print(f"\nMean reward: {results['mean']:.2f} ± {results['std']:.2f}")

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = f"./results/cartpole_results_dqn-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f)

    env.close()
    print("... finished")

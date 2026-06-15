from __future__ import print_function

import os
import gymnasium as gym
import json
from datetime import datetime
from agent.dqn_agent import DQNAgent
from train_carracing import run_episode
from agent.dqn_network_reinforcement import CNN
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v3", render_mode="human")

    history_length = 1

    # TODO: Define networks and load agent
    # ....

    Q = CNN(history_length=history_length, num_actions=5)
    Q_target = CNN(history_length=history_length, num_actions=5)
    agent = DQNAgent(
        Q=Q,
        Q_target=Q_target,
        num_actions=5,
        history_length=history_length,
    )
    agent.load("models/dqn_agent_carracing.pt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env,
            agent,
            deterministic=True,
            do_training=False,
            rendering=True,
            history_length=history_length,
            skip_frames=3,
        )
        episode_rewards.append(stats.episode_reward)
        print(f"Episode {i+1}: reward = {stats.episode_reward:.2f}")

    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    print(f"\nMean reward: {results['mean']:.2f} ± {results['std']:.2f}")

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = f"./results/carracing_results_dqn-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f)

    env.close()
    print("... finished")

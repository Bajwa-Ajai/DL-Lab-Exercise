import sys

sys.path.append(".")
from datetime import datetime
import numpy as np
import gymnasium as gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *
import torch
import torch.nn.functional as F

HISTORY_LENGTH = 8
N_CLASSES = 5


def preprocess_state(state, history, history_length):
    # RGB to grayscale conversion and normalization
    gray = rgb2gray(state) / 255.0

    history.append(gray)
    if len(history) > history_length:
        history.pop(0)

    while len(history) < history_length:
        history.insert(0, history[0])

    stacked = np.stack(history, axis=0)[np.newaxis].astype(np.float32)
    return stacked


def run_episode(env, agent, rendering=True, max_timesteps=1000):

    episode_reward = 0
    step = 0
    history = []

    state, _ = env.reset()

    if rendering:
        env.render()

    while True:
        X = preprocess_state(state, history, HISTORY_LENGTH)

        # Get action from agent using id_to_action from utils
        outputs = agent.predict(X)

        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
        action_id = int(probs.argmax())
        a = id_to_action(action_id)

        next_state, r, terminated, truncated, _ = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if terminated or truncated or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15

    # TODO: load agent
    # agent = BCAgent(...)
    # agent.load("models/bc_agent.pt")

    agent = BCAgent(history_length=HISTORY_LENGTH, n_classes=N_CLASSES)
    agent.load(f"models/bc_agent_history_length_{HISTORY_LENGTH}.pt")

    env = gym.make("CarRacing-v3", render_mode="human")

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)
        print(f"Episode {i+1}: reward = {episode_reward:.2f}")

    # Save results
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    print(f"\nMean reward: {results['mean']:.2f} ± {results['std']:.2f}")

    os.makedirs("results", exist_ok=True)
    fname = f"results/results_bc_agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f)

    env.close()
    print("... finished")

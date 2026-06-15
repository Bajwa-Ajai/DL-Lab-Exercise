# export DISPLAY=:0

# import pyglet
# pyglet.options["headless"] = True
import sys

sys.path.append("./")

import os
import numpy as np
import gymnasium as gym
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray
from utils import *
from agent.dqn_agent import DQNAgent
from agent.dqn_network_reinforcement import CNN


def run_episode(
    env,
    agent,
    deterministic,
    skip_frames=0,
    do_training=True,
    rendering=False,
    max_timesteps=1000,
    history_length=1,
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state, _ = env.reset()

    if rendering:
        env.render()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * history_length)
    state = np.array(image_hist)  # (history_length, 96, 96)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)

        # Hint: frame skipping might help you to get better results.
        action_id = agent.act(state, deterministic=deterministic)
        action = id_to_action(action_id)

        # Frame skipping: repeat action for skip_frames+1 steps and accumulate reward
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminated, truncated, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminated or truncated:
                break

        # Preprocess and update history
        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist)  # (history_length, 96, 96)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminated or truncated)

        stats.step(reward, action_id)
        state = next_state

        if terminated or truncated or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    num_eval_episodes=5,
    eval_cycle=20,
    history_length=1,
    max_timesteps=1000,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(
        tensorboard_dir,
        "CarRacing",
        stats=[
            "train/episode_reward",
            "train/straight",
            "train/left",
            "train/right",
            "train/accel",
            "train/brake",
            "eval/mean_episode_reward",
        ],
    )

    for i in range(1, num_episodes + 1):
        print("episode %d" % i)
        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        stats = run_episode(
            env,
            agent,
            max_timesteps=max_timesteps,
            deterministic=False,
            do_training=True,
            history_length=history_length,
            skip_frames=3,  # skip 3 frames — speeds up training significantly
        )

        # Decay epsilon
        agent.epsilon = max(0.05, agent.epsilon * 0.995)

        tensorboard.write_episode_data(
            i,
            eval_dict={
                "train/episode_reward": stats.episode_reward,
                "train/straight": stats.get_action_usage(STRAIGHT),
                "train/left": stats.get_action_usage(LEFT),
                "train/right": stats.get_action_usage(RIGHT),
                "train/accel": stats.get_action_usage(ACCELERATE),
                "train/brake": stats.get_action_usage(BRAKE),
            },
        )

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...

        # store model.
        if i % eval_cycle == 0:
            eval_rewards = []
            for j in range(num_eval_episodes):
                eval_stats = run_episode(
                    env,
                    agent,
                    deterministic=True,
                    do_training=False,
                    history_length=history_length,
                    skip_frames=3,
                )
                eval_rewards.append(eval_stats.episode_reward)
            mean_eval = np.mean(eval_rewards)
            print(f"  [eval] mean reward: {mean_eval:.2f}")
            tensorboard.write_episode_data(
                i, eval_dict={"eval/mean_episode_reward": mean_eval}
            )

        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent_carracing.pt"))

    tensorboard.close_session()


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20
    history_length = 1

    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    # TODO: Define Q network, target network and DQN agent
    # ...

    Q = CNN(history_length=history_length, num_actions=5)
    Q_target = CNN(history_length=history_length, num_actions=5)

    agent = DQNAgent(
        Q=Q,
        Q_target=Q_target,
        num_actions=5,
        gamma=0.95,
        batch_size=64,
        epsilon=0.9,  # start with high exploration for CarRacing
        tau=0.005,
        lr=1e-4,
        history_length=history_length,
    )

    train_online(
        env,
        agent,
        num_episodes=1000,
        num_eval_episodes=num_eval_episodes,
        eval_cycle=eval_cycle,
        history_length=history_length,
        max_timesteps=1000,
    )

    env.close()

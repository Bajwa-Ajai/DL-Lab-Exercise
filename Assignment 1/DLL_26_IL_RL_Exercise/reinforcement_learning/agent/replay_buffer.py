from collections import namedtuple, deque
import numpy as np


class ReplayBuffer:

    # Replay buffer with FIFO capacity for experience replay.
    def __init__(self, history_length=0, capacity=int(1e5)):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        Oldest entries are automatically dropped when capacity is exceeded (FIFO).
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def next_batch(self, batch_size):
        """
        This method samples a random batch of transitions.
        """
        batch_indices = np.random.choice(len(self.states), batch_size, replace=False)
        batch_states = np.array([self.states[i] for i in batch_indices])
        batch_actions = np.array([self.actions[i] for i in batch_indices])
        batch_next_states = np.array([self.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self.rewards[i] for i in batch_indices])
        batch_dones = np.array([self.dones[i] for i in batch_indices])
        return (
            batch_states,
            batch_actions,
            batch_next_states,
            batch_rewards,
            batch_dones,
        )

    def __len__(self):
        return len(self.states)

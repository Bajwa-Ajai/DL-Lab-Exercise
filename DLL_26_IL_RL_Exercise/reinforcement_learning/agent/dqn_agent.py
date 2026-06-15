import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        gamma=0.95,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-4,
        history_length=0,
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.
        """
        # setup networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q = Q.to(self.device)
        self.Q_target = Q_target.to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

    def train(self, state, action, next_state, reward, terminal):
        """
        Stores transition in replay buffer and performs a batch update.
        """
        # 1. Add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        # Only start training once the buffer has enough samples for a warm-up
        if len(self.replay_buffer) < 1000:
            return

        # 2. Sample a batch from replay buffer
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = (
            self.replay_buffer.next_batch(self.batch_size)
        )

        # Convert to tensors
        batch_states = torch.FloatTensor(batch_states).to(self.device)
        batch_actions = torch.LongTensor(batch_actions).to(self.device)
        batch_next_states = torch.FloatTensor(batch_next_states).to(self.device)
        batch_rewards = torch.FloatTensor(batch_rewards).to(self.device)
        batch_dones = torch.FloatTensor(batch_dones).to(self.device)

        # 2.1 Compute TD targets
        # td_target = reward + gamma * max_a Q_target(next_state, a)  (0 if terminal)
        with torch.no_grad():
            next_q_values = self.Q_target(batch_next_states)  # (B, num_actions)
            max_next_q = next_q_values.max(dim=1)[0]  # (B,)
            td_targets = batch_rewards + self.gamma * max_next_q * (1.0 - batch_dones)

        # Current Q values for the actions that were actually taken
        q_values = self.Q(batch_states)  # (B, num_actions)
        q_taken = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)  # (B,)

        # 2.2 Update Q network
        loss = self.loss_function(q_taken, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 2.3 Soft update target network
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        """
        Epsilon-greedy action selection.
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # Greedy action: argmax Q(state, a)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.Q(state_tensor)
            action_id = q_values.argmax(dim=1).item()
        else:
            # Random exploration — uniform for CartPole (2 actions)
            action_id = np.random.randint(0, self.num_actions)

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name, map_location=self.device))
        self.Q_target.load_state_dict(torch.load(file_name, map_location=self.device))

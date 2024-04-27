import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from attention import Attention

class QValueNet(torch.nn.Module):
    """
    A neural network for approximating Q-values with social attention mechanism for state inputs.
    This network integrates an attention mechanism to focus on relevant features of the input state.

    Parameters:
    state_dim (int): Dimension of the state space.
    hidden_dim (list): List of integers defining the dimensions of the hidden layers.
    action_dim (int): Number of possible actions.
    device (torch.device): The device tensors will be moved to.
    ego_dim (int, optional): Dimension of the ego input for the attention module.
    oppo_dim (int, optional): Dimension of the opponent input for the attention module.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, device, ego_dim=5, oppo_dim=5):
        super(QValueNet, self).__init__()
        self.attn = Attention(ego_dim, oppo_dim).to(device)
        layer1 = self.attn.embed_dim
        layer_dims = [layer1] + hidden_dim
        self.fc_layers = torch.nn.ModuleList(
            [torch.nn.Linear(layer_dims[i], layer_dims[i + 1]).to(device) for i in range(len(hidden_dim))]
        )
        self.fc_out = torch.nn.Linear(hidden_dim[-1], action_dim).to(device)

    def forward(self, x):
        """Forward pass for generating Q-values from state input using attention and fully connected layers."""
        if len(x.shape) > 2:
            x = self.attn(x[:, 0, :], x[:, :, :])
        else:
            x = self.attn(x[0], x[:])  # the first line is always the ego vehicle

        for layer in self.fc_layers:
            x = F.relu(layer(x))

        return self.fc_out(x)


class DQN(torch.nn.Module):
    """
    Implements a Double Deep Q-Network, which reduces the overestimation of Q-values
    by decoupling selection and evaluation of the action in the target network.

    Parameters:
    state_dim (int): Dimension of the state space.
    hidden_dim (list): List of hidden layer sizes.
    action_dim (int): Number of actions.
    learning_rate (float): Learning rate for the optimizer.
    gamma (float): Discount factor for future rewards.
    epsilon (float): Epsilon for the epsilon-greedy action selection.
    target_update (int): Frequency of update cycles for updating the target network.
    device (torch.device): Device to run the model's computations on.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        super(DQN, self).__init__()
        self.q_net = QValueNet(state_dim, hidden_dim, action_dim, device).to(device)
        self.target_q_net = QValueNet(state_dim, hidden_dim, action_dim, device).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):
        """Selects an action using epsilon-greedy strategy based on the current Q-value approximations."""
        if self.training and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # Creating a tensor from a list of numpy.ndarrays is extremely slow. 
            # So, we need to convert the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
            state = np.array([state])
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state_tensor).argmax().item()
        return action

    def max_q_value(self, state):
        """Returns the maximum Q-value for a given state."""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def eps_decay(self):
        """Decays epsilon value multiplicatively to reduce the exploration over time."""
        self.epsilon *= 0.9

    def update(self, transition_dict):
        """Performs a single update step on the Q-network based on a batch of transitions."""
        # Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        # So, we need to convert the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        states = np.array(transition_dict['states'])
        actions = np.array(transition_dict['actions'])
        rewards = np.array(transition_dict['rewards'])
        next_states = np.array(transition_dict['next_states'])
        dones = np.array(transition_dict['dones'])

        states_tensor = torch.tensor(states, dtype=torch.float).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).view(-1, 1).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states_tensor).gather(1, actions_tensor)
        next_actions = self.q_net(next_states_tensor).max(1)[1].view(-1, 1)  # Getting the max action indices
        max_next_q_values = self.target_q_net(next_states_tensor).gather(1, next_actions)
        q_targets = rewards_tensor + self.gamma * max_next_q_values * (1 - dones_tensor)
        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

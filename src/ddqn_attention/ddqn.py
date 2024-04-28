import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from .attention import Attention

class QValueNet(torch.nn.Module):
    """
    A neural network for approximating Q-values with social attention mechanism for state inputs.
    This network integrates an attention mechanism to focus on relevant features of the input state.

    Parameters:
    hidden_dimension (list): List of integers defining the dimensions of the hidden layers.
    action_dimension (int): Number of possible actions.
    device (torch.device): The device tensors will be moved to.
    ego_dim (int, optional): Dimension of the ego input for the attention module.
    oppo_dim (int, optional): Dimension of the opponent input for the attention module.
    """
    def __init__(self, hidden_dimension, action_dimension, device, ego_dim=5, oppo_dim=5):
        super(QValueNet, self).__init__()
        # Initialize the attention module, specifying dimensions for ego and opponent inputs.
        self.attn = Attention(ego_dim, oppo_dim).to(device)
        # Defines the input dimension for the first linear layer based on the attention embedding size.
        layer1 = self.attn.embed_dim

        # Assembles all layer dimensions starting with the output dimension of the attention layer.
        layer_dims = [layer1, ] + hidden_dimension

         # Creates a series of linear layers based on the dimensions in layer_dims.
        self.fc_layers = torch.nn.ModuleList(
            [torch.nn.Linear(layer_dims[i], layer_dims[i + 1]).to(device) for i in range(len(hidden_dimension))]
        )
        self.fc_out = torch.nn.Linear(hidden_dimension[-1], action_dimension).to(device)

    # @override
    def forward(self, x):
        """Forward pass for generating Q-values from state input using attention and fully connected layers."""

        # Handles different input shapes, applying attention based on the shape.
        if len(x.shape) > 2:
            x = self.attn(x[:, 0, :], x[:, :, :])
        else:
            x = self.attn(x[0], x[:])  # the first line is always the ego vehicle

        for layer in self.fc_layers:
            x = F.relu(layer(x))

        return self.fc_out(x)


class DDQN(torch.nn.Module):
    """
    Implements a Double Deep Q-Network, which reduces the overestimation of Q-values
    by decoupling selection and evaluation of the action in the target network.

    Parameters:
    action_dimension (int): Number of actions.
    hidden_dimension (list): List of hidden layer sizes.
    learning_rate (float): Learning rate for the optimizer.
    gamma (float): Discount factor for future rewards.
    epsilon (float): Epsilon for the epsilon-greedy action selection.
    target_update (int): Frequency of update cycles for updating the target network.
    device (torch.device): Device to run the model's computations on.
    """
    def __init__(
            self, 
            action_dimension = 5,
            hidden_dimension = [256, 256],
            learning_rate = 1e-3, 
            gamma=0.9, 
            epsilon=0.9, 
            target_update=60, 
            device = None,
        ):
        super(DDQN, self).__init__()
        
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.q_net = QValueNet(hidden_dimension, action_dimension, self.device).to(self.device)
        self.target_q_net = QValueNet(hidden_dimension, action_dimension, self.device).to(self.device)
        # Set up the optimizer and a scheduler for adjusting the learning rate.
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.action_dim = action_dimension
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0

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

    def update(self, transition_dict: dict):
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
        
        self.optimizer.zero_grad()

        # Calculation of loss
        loss = F.mse_loss(q_values, q_targets)
        loss.backward()
        
        # Apply gradients
        self.optimizer.step()

        # Update the learning rate
        self.scheduler.step()

        # Updates the target network every specified number of updates to stabilize training.
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

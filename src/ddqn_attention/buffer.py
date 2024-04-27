import collections
import random

class ReplayBuffer:
    """
    A class for storing game transitions (state, action, reward, next_state, done) in a cyclic buffer.
    The buffer supports adding transitions and sampling a batch of transitions for training purposes,
    often used in reinforcement learning algorithms to decorrelate successive samples by mixing them
    across episodes.

    Parameters:
    capacity (int): The maximum number of transitions the buffer can hold, at which point older
                    transitions are dropped as new ones are added.
    """

    def __init__(self, capacity):
        """
        Initializes the ReplayBuffer with a specific capacity.

        Args:
        capacity (int): The maximum number of transitions to store in the buffer.
        """
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Adds a transition to the replay buffer.

        Args:
        state: The state of the environment before taking the action.
        action: The action taken in the environment.
        reward: The reward received after taking the action.
        next_state: The state of the environment after taking the action.
        done: A boolean flag indicating if the episode ended after the action.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer.

        Args:
        batch_size (int): The number of transitions to sample.

        Returns:
        tuple of lists: (states, actions, rewards, next_states, dones), where each list contains elements
                        of the respective type drawn randomly from the buffer.
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):
        """
        Returns the current size of the internal buffer.

        Returns:
        int: The number of transitions currently stored in the buffer.
        """
        return len(self.buffer)

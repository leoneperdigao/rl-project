import logging

from tqdm import tqdm
import numpy as np
import gymnasium as gym

from .ddqn import DDQN
from .buffer import ReplayBuffer

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Function to use tqdm.write to avoid messing up tqdm progress bars
def tqdm_logging_handler(msg, *args):
    tqdm.write(msg % args, end='')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(stream=tqdm_logging_handler)]
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train(
        env: gym.Env, 
        agent: DDQN, 
        num_episodes: int,
        replay_buffer: ReplayBuffer, 
        minimal_size: int, 
        batch_size: int,
        reward_scaling_factor: float = 10,
        with_rendering: bool = False,
    ):
    """
    Trains an off-policy reinforcement learning agent using the given environment, replay buffer, and agent configuration.

    Args:
    env (gym.Env): The environment object which follows the OpenAI Gym interface.
    agent (DDQN): The agent to be trained which supports take_action, update, and scheduler methods.
    num_episodes (int): Total number of episodes for training.
    replay_buffer: The ReplayBuffer object for storing and sampling transitions.
    minimal_size (int): The minimum buffer size before starting training updates.
    batch_size (int): The number of samples per batch for updates.

    Returns:
    list: A list of returns for each training iteration.
    """
    results = []
    tqdm.write(f"Starting training: {num_episodes} episodes")
    pbar = tqdm(total=num_episodes, desc='Total Progress')

    for episode_number in range(num_episodes):
        episode_reward = 0
        state, _ = env.reset()
        done = truncated = False
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            reward *= reward_scaling_factor  # Reward scaling for better training performance
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if replay_buffer.size() > minimal_size and replay_buffer.size() >= batch_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict)
            
            if with_rendering:
                env.render()

        results.append(episode_reward)
        pbar.update(1)
        pbar.set_postfix({'episode': f'{episode_number + 1}', 'mean_reward': f'{np.mean(results[-10:]):.3f}'})

        if (episode_number + 1) % 100 == 0 or episode_number + 1 == num_episodes:
            agent.eps_decay()
            logger.debug(f"After {episode_number + 1} episodes, current epsilon: {agent.epsilon}")

    pbar.close()
    tqdm.write("Training complete.")
    return results


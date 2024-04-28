from tqdm import tqdm
import numpy as np


def moving_average(a, window_size):
    """
    Calculates the moving average of the provided array using the specified window size.

    Args:
    a (np.array): Input array.
    window_size (int): The number of elements to consider for each moving average calculation.

    Returns:
    np.array: The moving average of the array.
    """
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    """
    Trains an off-policy reinforcement learning agent using the given environment, replay buffer, and agent configuration.

    Args:
    env: The environment object which follows the OpenAI Gym interface.
    agent: The agent to be trained which supports take_action, update, and scheduler methods.
    num_episodes (int): Total number of episodes for training.
    replay_buffer: The ReplayBuffer object for storing and sampling transitions.
    minimal_size (int): The minimum buffer size before starting training updates.
    batch_size (int): The number of samples per batch for updates.

    Returns:
    list: A list of returns for each training iteration.
    """
    return_list = []
    for i in range(100):  # Divide the training into 100 separate progress bars for clarity
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                # state = filter_non_presence(state)
                done = truncated = False
                while not (done or truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    reward *= 10  # Reward scaling for better training performance
                    # next_state = filter_non_presence(next_state)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': f'{num_episodes/10 * i + i_episode + 1}', 'return': f'{np.mean(return_list[-10:]):.3f}'})
                pbar.update(1)
        agent.scheduler.step()
        try:
            agent.eps_decay()
            print("Probability of random exploration (epsilon): ", agent.epsilon)
        except:
            pass
    return return_list


def train_off_policy_agent_with_rendering(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    """
    Similar to train_off_policy_agent but includes rendering of the environment, which is useful for visual feedback during training.

    The parameters are the same as train_off_policy_agent. This function is specifically for environments that support rendering.
    """
    return_list = []
    for i in range(100):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = truncated = False
                while not (done or truncated):
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    reward *= 10
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                    env.render()
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': f'{num_episodes/10 * i + i_episode + 1}', 'return': f'{np.mean(return_list[-10:]):.3f}'})
                pbar.update(1)
    return return_list

import numpy as np


def moving_average(input_array, window_size):
    """
    Calculates the moving average of the provided array using the specified window size.

    Args:
    input_array (np.array): Input array.
    window_size (int): The number of elements to consider for each moving average calculation.

    Returns:
    np.array: The moving average of the array.
    """
    cumulative_sum = np.cumsum(np.insert(input_array, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size, 2)
    begin = np.cumsum(input_array[:window_size-1])[::2] / r
    end = (np.cumsum(input_array[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
import xml.etree.ElementTree as ET
import scipy as sp
import h5py
from scipy.ndimage import gaussian_filter1d
import numpy as np

def get_trial_id(labels,upper_threshold, lower_threshold):
    signal = labels

    # Array to store cycle IDs
    cycle_ids = np.zeros_like(signal, dtype=int)

    # Variables to keep track of the state and current cycle ID
    current_cycle = 0
    cycle_started = False
    above_threshold = False

    # Iterate over the signal
    for i in range(len(signal)):
        if not cycle_started and signal[i] < lower_threshold:
            # Check if the signal is around zero to start a cycle
            cycle_started = True
            current_cycle += 1

        if cycle_started:
            # Check if the signal has crossed the upper threshold
            if signal[i] > upper_threshold:
                above_threshold = True

            # Check if the signal returns to around zero and it was above the threshold
            if above_threshold and signal[i] < lower_threshold:
                cycle_started = False  # End of cycle
                above_threshold = False  # Reset for the next cycle

        # Assign the current cycle ID
        cycle_ids[i] = current_cycle
    return cycle_ids


def get_directions(signal, low_threshold, high_threshold):
    """
    Assigns a direction to each point in the signal based on threshold crossings.

    Parameters:
    - signal: np.array, the input signal data.
    - low_threshold: float, the lower boundary for detecting upward trends.
    - high_threshold: float, the upper boundary for detecting downward trends.

    Returns:
    - directions: np.array, an array containing 1 for upward trend, -1 for downward trend,
                  and 0 for no clear direction.
    """
    # Initialize directions with zeros
    directions = np.zeros_like(signal)

    # Track the current direction
    current_direction = 0

    # Iterate through the signal
    for i in range(1, len(signal)):
        if signal[i] < high_threshold and signal[i - 1] >= high_threshold:
            current_direction = -1
        elif signal[i] > low_threshold and signal[i - 1] <= low_threshold:
            current_direction = 1
        if signal[i] < low_threshold:
            current_direction = 0
        if signal[i] > high_threshold:
            current_direction = 0

        # Assign the current direction
        directions[i] = current_direction

    return directions


def get_speed(signal, time_stamps, filter_size=10):
    """
    Computes the derivative of the signal (speed) and applies a Gaussian filter.

    Parameters:
    - signal: np.array, the input signal data.
    - filter_size: int, the sigma for the Gaussian kernel, controlling the amount of smoothing.

    Returns:
    - filtered_speed: np.array, the speed of the signal, smoothed, and same length as input signal.
    """
    # Calculate the speed as the difference between consecutive elements
    speed_pos = np.diff(signal)
    timediff = np.diff(time_stamps)
    speed = speed_pos / timediff
    # To make speed the same length as signal, append the last computed difference to the end
    speed = np.append(speed, speed[-1])

    # Apply a Gaussian filter to the speed
    filtered_speed = gaussian_filter1d(speed, sigma=filter_size)

    return filtered_speed


def compute_internal_trial_time(trial_ids, sampling_rate):
    """
    Computes an internal trial time that resets to zero at the start of each new trial.

    Parameters:
    - trial_ids: np.array, an array of trial IDs where an increment indicates a new trial.
    - sampling_rate: float, the rate at which data is sampled per second.

    Returns:
    - trial_time: np.array, a time array that resets to zero at the beginning of each trial.
    """
    # Initialize the trial time array
    trial_time = np.zeros_like(trial_ids, dtype=float)

    # Calculate time step per sample
    time_step = 1 / sampling_rate

    # Track the start of a new trial
    current_trial_id = trial_ids[0]
    last_reset_index = 0

    # Iterate through the trial IDs
    for i in range(1, len(trial_ids)):
        if trial_ids[i] != current_trial_id:
            # A new trial has started
            current_trial_id = trial_ids[i]
            last_reset_index = i
        # Calculate time since the last trial start
        trial_time[i] = (i - last_reset_index) * time_step

    return trial_time

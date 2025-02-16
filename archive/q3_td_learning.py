"""Corrected TD learning simulation for Exercise 3 but misses box car feature type"""

import numpy as np
import matplotlib.pyplot as plt

# ENVIRONMENT #
def stimulus_function(t):
    return 1.0 if np.isclose(t, 10.0) else 0.0

def reward_function(t):
    
    mean = 20.0 # reward at 20s
    std = 1.0 # given my exercise
    
    return 0.5 * np.exp(-((t - mean) ** 2) / (2 * (std ** 2)))

# LEARNING PARAMS #
def get_state(history):
    return np.array(history)
    
def get_value(state, weights):
    return np.sum(state * weights)

def get_td_error(state, next_state, reward, weights, gamma):
    current_value = get_value(state, weights)
    next_value = get_value(next_state, weights)
    td_error = reward + gamma * next_value - current_value

    return td_error


# SIMULATION RUN #
def run_simulation(n_trials=201, t_max=25.0, dt=0.5, memory_span=12.0, epsilon=0.1, gamma=1.0):
    """Run the TD learning simulation"""
    
    # Initialize parameters
    t_steps = int(t_max / dt)
    memory_steps = int(memory_span / dt)
    time_points = np.arange(0, t_max, dt)
    
    # Initialize weights
    weights = np.zeros(memory_steps)
    
    # Storage for plotting
    values_history = np.zeros((n_trials, t_steps))
    td_errors_history = np.zeros((n_trials, t_steps))
    value_differences = np.zeros((n_trials, t_steps))
    
    for trial in range(n_trials):
        
        # Initialize state history for new trial
        state_history = [0.0] * memory_steps # (Python list)
        current_state = get_state(state_history)
        
        # Storage for current trial
        trial_values = np.zeros(t_steps)
        trial_td_errors = np.zeros(t_steps)
        
        for t_idx, t in enumerate(time_points):
            
            # Get stimulus and reward
            stimulus = stimulus_function(t)
            reward = reward_function(t)
            
            # Move state history forward and get next state
            state_history = state_history[1:] + [stimulus]
            next_state = get_state(state_history)
            
            # Calculate TD error to get compute next weight update and move on to next state
            td_error = get_td_error(current_state, next_state, reward, weights, gamma)
            
            #############################################
            weights += epsilon * td_error * current_state
            #############################################
            
            current_state = next_state
            
            # Store values for plot
            value = get_value(current_state, weights)
            trial_values[t_idx] = value
            trial_td_errors[t_idx] = td_error
        
        values_history[trial] = trial_values
        td_errors_history[trial] = trial_td_errors
        
        # Calculate value differences
        value_differences[trial, 1:] = gamma * trial_values[1:] - trial_values[:-1]
    
    return time_points, values_history, value_differences, td_errors_history

def plot_results(time_points, values_history, value_differences, td_errors_history):
    """Plot the results"""
    # Select trials to plot (every 10th trial)
    trial_indices = np.arange(0, values_history.shape[0], 10)
    
    # Create figure with extra width for legend
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(8, 11), sharex=True)
    
    # Get colors from viridis colormap (reversed so most recent trials are violet)
    colors = plt.cm.winter(np.linspace(0, 1, len(trial_indices)))[::-1]
    
    # Plot stimulus 
    ax0.plot(time_points, [stimulus_function(t) for t in time_points], color="black", label='Stimulus', alpha=0.5)
    ax0.set_ylabel('')
    ax0.set_title('(a) Stimulus')
    
    # Plot reward
    ax1.plot(time_points, [reward_function(t) for t in time_points], color="black", label='Reward', alpha=0.5)
    ax1.set_ylabel('')
    ax1.set_title('(b) Reward')
    
    # Plot values
    for idx, color in zip(trial_indices, colors):
        ax2.plot(time_points, values_history[idx], color=color, alpha=0.8, label=f'Trial {idx+1}')
    ax2.set_ylabel('')
    ax2.set_title('(c) Value Function')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Plot value differences
    for idx, color in zip(trial_indices, colors):
        ax3.plot(time_points, value_differences[idx], color=color, alpha=0.8)
    ax3.set_ylabel('')
    ax3.set_title('(d) Temporal Difference of Value')
    
    # Plot TD errors
    for idx, color in zip(trial_indices, colors):
        ax4.plot(time_points, td_errors_history[idx], color=color, alpha=0.8)
    ax4.set_xlabel('')
    ax4.set_ylabel('')
    ax4.set_title('(e) TD Learning Error')
    
    #plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(f'results/question1/ex3_td_learning.png')
    plt.show()

# Run simulation and plot results
time_points, values_history, value_differences, td_errors_history = run_simulation()
plot_results(time_points, values_history, value_differences, td_errors_history)
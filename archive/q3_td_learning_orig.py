"""Output from Claude for exercise 3"""

import numpy as np
import matplotlib.pyplot as plt

def stimulus_function(t):
    """Stimulus function - spike at t=10s"""
    return 1.0 if abs(t - 10.0) < 0.25 else 0.0

def reward_function(t):
    """Reward function - Gaussian around t=20s"""
    return 0.5 * np.exp(-((t - 20.0) ** 2) / (2 * 1.0 ** 2))

def get_state(history):
    """Convert stimulus history to state representation"""
    return np.array(history)

def get_value(state, weights):
    """Compute value from state using current weights"""
    return np.sum(state * weights)

def update_weights(weights, state, next_state, reward, epsilon, gamma):
    """Update weights using TD learning"""
    current_value = get_value(state, weights)
    next_value = get_value(next_state, weights)
    td_error = reward + gamma * next_value - current_value
    weights += epsilon * td_error * state
    return current_value, td_error

def run_simulation(n_trials=201, t_max=25.0, dt=0.5, memory_span=12.0, epsilon=0.2, gamma=1.0):
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
    
    # Run trials
    for trial in range(n_trials):
        # Initialize state history
        state_history = [0.0] * memory_steps
        current_state = get_state(state_history)
        
        # Storage for current trial
        trial_values = np.zeros(t_steps)
        trial_td_errors = np.zeros(t_steps)
        
        # Run through time steps
        for t_idx, t in enumerate(time_points):
            # Get stimulus and reward
            stimulus = stimulus_function(t)
            reward = reward_function(t)
            
            # Update state history
            state_history = state_history[1:] + [stimulus]
            next_state = get_state(state_history)
            
            # Get value and update weights
            value, td_error = update_weights(weights, current_state, next_state, 
                                           reward, epsilon, gamma)
            
            # Store results
            trial_values[t_idx] = value
            trial_td_errors[t_idx] = td_error
            
            # Update state
            current_state = next_state
        
        # Store trial results
        values_history[trial] = trial_values
        td_errors_history[trial] = trial_td_errors
        
        # Calculate value differences
        value_differences[trial, 1:] = gamma * trial_values[1:] - trial_values[:-1]
    
    return time_points, values_history, value_differences, td_errors_history

def plot_results(time_points, values_history, value_differences, td_errors_history):
    """Plot the results"""
    # Select trials to plot (every 10th trial)
    trial_indices = np.arange(0, values_history.shape[0], 10)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot stimulus and reward
    ax0 = ax1.twinx()
    ax0.plot(time_points, [stimulus_function(t) for t in time_points], 'r--', 
             label='Stimulus', alpha=0.5)
    ax0.plot(time_points, [reward_function(t) for t in time_points], 'g--', 
             label='Reward', alpha=0.5)
    ax0.legend()
    
    # Plot values
    for idx in trial_indices:
        ax1.plot(time_points, values_history[idx], alpha=0.5, label=f'Trial {idx+1}')
    ax1.set_ylabel('Value V(t)')
    ax1.set_title('(a) Value Function')
    
    # Plot value differences
    for idx in trial_indices:
        ax2.plot(time_points, value_differences[idx], alpha=0.5)
    ax2.set_ylabel('Value Difference ΔV(t)')
    ax2.set_title('(b) Temporal Difference of Value')
    
    # Plot TD errors
    for idx in trial_indices:
        ax3.plot(time_points, td_errors_history[idx], alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('TD Error δ(t)')
    ax3.set_title('(c) TD Learning Error')
    
    plt.tight_layout()
    plt.show()

# Run simulation and plot results
time_points, values_history, value_differences, td_errors_history = run_simulation()
plot_results(time_points, values_history, value_differences, td_errors_history)
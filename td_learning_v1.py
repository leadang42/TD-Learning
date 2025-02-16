import numpy as np
import matplotlib.pyplot as plt

# ENVIRONMENT #
def stimulus_function(t):
    return 1.0 if np.isclose(t, 10.0) else 0.0

def reward_function(t):
    mean = 20.0  # reward at 20s
    std = 1.0    # given in exercise
    return 0.5 * np.exp(-((t - mean) ** 2) / (2 * (std ** 2)))

def get_state(history, feature_detector="tapped_delay_line"):
    if feature_detector == "tapped_delay_line":
        # Each feature is just the stimulus value at that time delay
        return np.array(history)
    
    elif feature_detector == "box_car":
        # Each feature is the sum of stimulus values up to that time delay
        box_car_state = np.zeros(len(history))
        for i in range(len(history)):
            # Sum all stimulus values from current time to tau steps in the past
            box_car_state[i] = sum(history[-(i+1):])
        return box_car_state

def get_value(state, weights):
    return np.sum(state * weights)

def get_td_error(state, next_state, reward, weights, gamma):
    current_value = get_value(state, weights)
    next_value = get_value(next_state, weights)
    td_error = reward + gamma * next_value - current_value
    return td_error

# SIMULATION RUN #
def run_simulation(feature_detector="tapped_delay_line", n_trials=201, t_max=25.0, dt=0.5, 
                  memory_span=12.0, epsilon=0.2, gamma=1.0):
    
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
        stimulus_history = [0.0] * memory_steps  # (Python list)
        current_state = get_state(stimulus_history, feature_detector)
        
        # Storage for current trial
        trial_values = np.zeros(t_steps)
        trial_td_errors = np.zeros(t_steps)
        
        # Create array to store states for this trial
        trial_states = np.zeros((t_steps, memory_steps))
        
        for t_idx, t in enumerate(time_points):
            # Get stimulus and reward
            stimulus = stimulus_function(t)
            reward = reward_function(t)
            
            # Move state history forward and get next state
            stimulus_history = stimulus_history[1:] + [stimulus]
            next_state = get_state(stimulus_history, feature_detector)
            
            # Store current state
            trial_states[t_idx] = current_state
            
            # Calculate TD error and update weights
            td_error = get_td_error(current_state, next_state, reward, weights, gamma)
            weights += epsilon * td_error * current_state
            
            current_state = next_state
            
            # Store values for plot
            value = get_value(current_state, weights)
            trial_values[t_idx] = value
            trial_td_errors[t_idx] = td_error
        
        values_history[trial] = trial_values
        td_errors_history[trial] = trial_td_errors
        
        # Calculate value differences
        value_differences[trial, 1:] = gamma * trial_values[1:] - trial_values[:-1]
        
        # Save states for this trial
        filename = f'states_{feature_detector}.txt'
        np.savetxt(filename, trial_states, fmt='%.6f', delimiter=',')
    
    return time_points, values_history, value_differences, td_errors_history

def plot_results(time_points, values_history, value_differences, td_errors_history, feature_detector):
    # Create figure with larger size
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    
    # Select trials to plot (every 10th trial)
    trial_indices = np.arange(0, values_history.shape[0], 10)
    
    # Get colors from winter colormap (reversed gradient)
    colors = plt.cm.winter(np.linspace(1, 0, len(trial_indices)))
    
    # Plot stimulus and reward
    axes[0].plot(time_points, [stimulus_function(t) for t in time_points], 
                color=plt.cm.winter(0.0), label='Stimulus', linewidth=2)
    axes[0].plot(time_points, [reward_function(t) for t in time_points], 
                color=plt.cm.winter(1.0), label='Reward', linewidth=2)
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Stimulus and Reward')
    axes[0].legend(loc='upper right')
    
    # Create legend elements for selected trials
    legend_indices = [0, len(trial_indices)//2, len(trial_indices)-1]
    legend_elements = []
    
    # Plot values
    for i, (idx, color) in enumerate(zip(trial_indices, colors)):
        line = axes[1].plot(time_points, values_history[idx], color=color, 
                          alpha=0.7, linewidth=1.5, label=f'Trial {idx+1}')
        if i in legend_indices:
            legend_elements.append(line[0])
    
    axes[1].set_ylabel('Value Estimate')
    axes[1].set_title('Value Function Over Learning')
    axes[1].legend(handles=legend_elements, 
                  labels=[f'Trial {trial_indices[i]+1}' for i in legend_indices], 
                  loc='upper right')
    
    # Plot value differences
    for idx, color in zip(trial_indices, colors):
        axes[2].plot(time_points, value_differences[idx], color=color, 
                    alpha=0.7, linewidth=1.5)
    axes[2].set_ylabel('ΔV(t)')
    axes[2].set_title('Temporal Difference of Value Function')
    
    # Plot TD errors
    for idx, color in zip(trial_indices, colors):
        axes[3].plot(time_points, td_errors_history[idx], color=color, 
                    alpha=0.7, linewidth=1.5)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('TD Error δ(t)')
    axes[3].set_title('TD Error Over Learning')
    
    plt.tight_layout()
    plt.savefig(f'results/td_learning_{feature_detector}.png')
    plt.show()

def run_both_simulations():
    """Run simulations for both tapped delay line and boxcar features."""
    
    # Run tapped delay line simulation
    print("\nRunning Tapped Delay Line simulation...")
    time_points, values_history, value_differences, td_errors_history = run_simulation(
        feature_detector="tapped_delay_line",
        epsilon=0.1, 
        gamma=1.0, 
        memory_span=12.0
    )
    plot_results(time_points, values_history, value_differences, td_errors_history, "tapped_delay_line")
    
    # Run boxcar simulation
    print("\nRunning Boxcar simulation...")
    time_points, values_history, value_differences, td_errors_history = run_simulation(
        feature_detector="box_car",
        epsilon=0.1, 
        gamma=1.0, 
        memory_span=12.0
    )
    plot_results(time_points, values_history, value_differences, td_errors_history, "box_car")

if __name__ == "__main__":
    run_both_simulations()
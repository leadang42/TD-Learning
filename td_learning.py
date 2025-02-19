import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# TD LEARNING FUNCTIONS #
def stimulus_function(t):
    return 1.0 if np.isclose(t, 10.0) else 0.0

def reward_function(t, reward_present=True):
    """Reward function with option to omit reward"""
    if not reward_present:
        return 0.0
    mean = 20.0
    std = 1.0
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

def td_learning(p=0.5, n_trials=1000, feature_detector="box_car", t_max=25.0, dt=0.5, memory_span=12.0, epsilon=0.01, gamma=1.0):
    """Run TD learning with partial reinforcement probability p"""
    
    # Initialize parameters
    t_steps = int(t_max / dt)
    memory_steps = int(memory_span / dt)
    time_points = np.arange(0, t_max, dt)
    
    # Initialize weights
    weights = np.zeros(memory_steps)
    
    # Storage for analysis
    values_history = np.zeros((n_trials, t_steps))
    td_errors_history = np.zeros((n_trials, t_steps))
    value_differences = np.zeros((n_trials, t_steps))
    
    reward_present = np.zeros(n_trials, dtype=bool)
    
    for trial in range(n_trials):
        # Determine if reward will be present this trial
        reward_present[trial] = np.random.random() < p
        
        # Initialize state history
        stimulus_history = [0.0] * memory_steps
        current_state = get_state(stimulus_history, feature_detector)
        
        # Storage for current trial
        trial_values = np.zeros(t_steps)
        trial_td_errors = np.zeros(t_steps)
        
        for t_idx, t in enumerate(time_points):
            # Get stimulus and reward
            stimulus = stimulus_function(t)
            reward = reward_function(t, reward_present[trial])
            
            # Update state history and get next state
            stimulus_history = stimulus_history[1:] + [stimulus]
            next_state = get_state(stimulus_history, feature_detector)
            
            # Calculate TD error and update weights
            td_error = get_td_error(current_state, next_state, reward, weights, gamma)
            weights += epsilon * td_error * current_state
            
            current_state = next_state
            
            # Store values
            value = get_value(current_state, weights)
            trial_values[t_idx] = value
            trial_td_errors[t_idx] = td_error
            
        values_history[trial] = trial_values
        td_errors_history[trial] = trial_td_errors
        value_differences[trial, 1:] = gamma * trial_values[1:] - trial_values[:-1]
    
    return time_points, values_history, value_differences, td_errors_history, reward_present

# DOPAMINE FUNCTIONS #
def model_dopamine(x, alpha=6.0, beta=6.0, x_star=0.27):
    """Model dopamine response with limited dynamic range"""
    if x < 0:
        return x/alpha
    elif x < x_star:
        return x
    else:
        return x_star + (x - x_star)/beta

# EXERCISE 3 AND 4 #
def plot_tdl_vars(time_points, values_history, value_differences, td_errors_history, feature_detector):

    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    trial_indices = np.arange(0, values_history.shape[0], 10)
    colors = plt.cm.viridis(np.linspace(1, 0, len(trial_indices))*0.8)
    
    # Plot stimulus and reward
    axes[0].plot(time_points, [stimulus_function(t) for t in time_points], color=plt.cm.viridis(0.0), label='Stimulus', linewidth=2)
    axes[0].plot(time_points, [reward_function(t) for t in time_points], color=plt.cm.viridis(0.8), label='Reward', linewidth=2)
    axes[0].set_ylabel('')
    axes[0].set_title('Stimulus and Reward')
    axes[0].legend(loc='upper right')
    
    legend_indices = [0, len(trial_indices)//2, len(trial_indices)-1]
    legend_elements = []
    
    # Plot values with legend
    for i, (idx, color) in enumerate(zip(trial_indices, colors)):
        line = axes[1].plot(time_points, values_history[idx], color=color, alpha=0.7, linewidth=1.5, label=f'Trial {idx+1}')
        
        if i in legend_indices:
            legend_elements.append(line[0])
    
    axes[1].set_ylabel('V')
    axes[1].set_title('Value Estimate Over Learning')
    axes[1].legend(handles=legend_elements, labels=[f'Trial {trial_indices[i]+1}' for i in legend_indices], loc='upper right')
    
    # Plot value differences
    for idx, color in zip(trial_indices, colors):
        axes[2].plot(time_points, value_differences[idx], color=color, alpha=0.7, linewidth=1.5)
    axes[2].set_ylabel('ΔV')
    axes[2].set_title('Temporal Difference of Value Estimate')
    
    # Plot TD errors
    for idx, color in zip(trial_indices, colors):
        axes[3].plot(time_points, td_errors_history[idx], color=color, alpha=0.7, linewidth=1.5)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('δ(t)')
    axes[3].set_title('TD Error Over Learning')
    
    plt.tight_layout()
    plt.savefig(f'results/td_learning_{feature_detector}.png')
    plt.show()

def exercise_3():
    # Run tapped delay line simulation
    print("\nRunning Tapped Delay Line simulation...")
    time_points, values_history, value_differences, td_errors_history, _ = td_learning(
        p=1.0,
        n_trials=201,
        feature_detector="tapped_delay_line",
        epsilon=0.2, 
        gamma=1.0, 
        memory_span=12.0
    )
    plot_tdl_vars(time_points, values_history, value_differences, td_errors_history, "tapped_delay_line")

def exercise_4():    
    # Run boxcar simulation
    print("\nRunning Boxcar simulation...")
    time_points, values_history, value_differences, td_errors_history, _ = td_learning(
        p=1.0,
        n_trials=121,
        feature_detector="box_car",
        epsilon=0.01, 
        gamma=1.0, 
        memory_span=12.0
    )
    plot_tdl_vars(time_points, values_history, value_differences, td_errors_history, "box_car")

# EXERCISE 5 #
def plot_tdl_vars_mean(time_points, mean_100trials, p):
    """Plot results from partial reinforcement analysis"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot value estimates
    axes[0].plot(time_points, mean_100trials['values'][2], color=plt.cm.viridis(0.8), label='Unrewarded')
    axes[0].plot(time_points, mean_100trials['values'][1], color=plt.cm.viridis(0.5), label='Rewarded')
    axes[0].plot(time_points, mean_100trials['values'][0], color=plt.cm.viridis(0.0), label='All trials')
    axes[0].set_ylabel('Value Estimate')
    axes[0].set_title(f'Value Estimate')
    axes[0].legend()
    
    # Plot value differences
    axes[1].plot(time_points, mean_100trials['value_differences'][2], color=plt.cm.viridis(0.8), label='Unrewarded')
    axes[1].plot(time_points, mean_100trials['value_differences'][1], color=plt.cm.viridis(0.5), label='Rewarded')
    axes[1].plot(time_points, mean_100trials['value_differences'][0], color=plt.cm.viridis(0.0), label='All trials')
    axes[1].set_ylabel('Value Differences')
    axes[1].set_title('Value Differences')
    axes[1].legend()
    
    # Plot TD errors
    axes[2].plot(time_points, mean_100trials['td_errors'][2], color=plt.cm.viridis(0.8), label='Unrewarded')
    axes[2].plot(time_points, mean_100trials['td_errors'][1], color=plt.cm.viridis(0.5), label='Rewarded')
    axes[2].plot(time_points, mean_100trials['td_errors'][0], color=plt.cm.viridis(0.0), label='All trials')
    axes[2].set_ylabel('TD Error')
    axes[2].set_title('TD Error')
    axes[2].legend()
    
    # Plot dopamine signal
    #axes[3].plot(time_points, mean_100trials['dopamine'], color='black')
    #axes[3].set_xlabel('Time (s)')
    #axes[3].set_ylabel('DA(δ(t))')
    #axes[3].set_title('Modeled Dopamine Signal')
    
    plt.tight_layout()
    plt.savefig(f'results/partial_reinforcement_p{p}.png')
    plt.show()
    
    return fig

def plot_dopamine_activity(time_points, dopamine_activities, p_values):
    plt.figure(figsize=(12, 5))
    
    for i, p in enumerate(p_values):
        plt.plot(time_points, dopamine_activities[i], color=plt.cm.viridis((1-p)*0.8), label=f'p={p}')
        
    #plt.plot(time_points, dopamine_activities[0], color=plt.cm.viridis(0.0))
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('DA(δ(t))')
    plt.title('Modeled Dopamine Signal')
    
    plt.savefig(f'results/dopamine_activity_p{p_values}.png')
    plt.show()
    
    return plt.gcf()

def compute_mean_100trials(time_points, values_history, value_differences, td_errors_history, reward_present):
    """Analyze results from last 100 trials"""
    
    # Get last 100 trials
    last_100_values = values_history[-100:, :]
    last_100_value_differences = value_differences[-100:, :]
    last_100_td_errors = td_errors_history[-100:, :]
    last_100_reward = reward_present[-100:]
    
    # Separate rewarded and unrewarded trials
    rewarded_values = last_100_values[last_100_reward]
    unrewarded_values = last_100_values[~last_100_reward]
    rewarded_value_differences = last_100_value_differences[last_100_reward]
    unrewarded_value_differences = last_100_value_differences[~last_100_reward]
    rewarded_td_errors = last_100_td_errors[last_100_reward]
    unrewarded_td_errors = last_100_td_errors[~last_100_reward]
    
    # Calculate means
    mean_all_values = np.mean(last_100_values, axis=0)
    mean_rewarded_values = np.mean(rewarded_values, axis=0)
    mean_unrewarded_values = np.mean(unrewarded_values, axis=0)
    
    mean_all_value_differences = np.mean(last_100_value_differences, axis=0)
    mean_rewarded_value_differences = np.mean(rewarded_value_differences, axis=0)
    mean_unrewarded_value_differences = np.mean(unrewarded_value_differences, axis=0)
    
    mean_all_td_errors = np.mean(last_100_td_errors, axis=0)
    mean_rewarded_td_errors = np.mean(rewarded_td_errors, axis=0)
    mean_unrewarded_td_errors = np.mean(unrewarded_td_errors, axis=0)
    
    # Convert TD errors to dopamine signal
    # dopamine_signal = np.vectorize(model_dopamine)(mean_all_td_errors)
    mean_dopamine_signal = np.mean(np.vectorize(model_dopamine)(last_100_td_errors), axis=0)
    
    #dopamine_signal = np.zeros_like(last_100_td_errors[0]) 
    
    #for trial_idx in range(len(last_100_td_errors)):
        
    #    for time_idx in range(len(last_100_td_errors[trial_idx])):
    #        dopamine_signal[time_idx] += model_dopamine(last_100_td_errors[trial_idx][time_idx])
    
    #mean_dopamine_signal = dopamine_signal / len(last_100_td_errors)
    
    return {
        'values': (mean_all_values, mean_rewarded_values, mean_unrewarded_values),
        'value_differences': (mean_all_value_differences, mean_rewarded_value_differences, mean_unrewarded_value_differences),
        'td_errors': (mean_all_td_errors, mean_rewarded_td_errors, mean_unrewarded_td_errors),
        'dopamine': mean_dopamine_signal
    }
    
def exercise_5(): 
    print("\nRunning partial reinforcement simulation (p=0.5)...")
    time_points, values_history, value_differences, td_errors_history, reward_present = td_learning(p=0.5)
    mean_100trials = compute_mean_100trials(time_points, values_history, value_differences, td_errors_history, reward_present)
    
    plot_tdl_vars_mean(time_points, mean_100trials, p=0.5)

    plot_dopamine_activity(time_points, [mean_100trials['dopamine']], p_values=[0.5])
   
def exercise_6_7():
    # Exercise 7-8: Analyze multiple reward probabilities
    print("\nAnalyzing different reward probabilities...")
    p_values = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    n_trials=1000
    feature_detector="box_car"
    epsilon=0.01
    t_max=25.0
    dt=0.5
    
    stimulus_peaks = []
    reward_peaks = []
    dopamine_activities = []
    
    for p in p_values:
        
        time_points, values_history, values_differences_history, td_errors_history, reward_present = td_learning(p=p, n_trials=n_trials, feature_detector=feature_detector, epsilon=epsilon, t_max=t_max, dt=dt)
        analysis = compute_mean_100trials(time_points, values_history, values_differences_history, td_errors_history, reward_present)
        
        # Find peaks around stimulus (10s) and reward (20s) times
        stimulus_idx = int(10.0 / dt)
        reward_idx = int(20.0 / dt)
        window_idx = int(2.5 / dt)
        
        dopamine = analysis['dopamine']
        dopamine_activities.append(deepcopy(dopamine))
        
        max_dopamine_at_stimulus = np.max(dopamine[stimulus_idx-window_idx:stimulus_idx+window_idx])
        max_dopamine_at_reward = np.max(dopamine[reward_idx-window_idx:reward_idx+window_idx])
        
        # Get dopamine levels at these times
        stimulus_peaks.append(max_dopamine_at_stimulus)
        reward_peaks.append(max_dopamine_at_reward)

    # Plot dopamine activity for different reward probabilities
    plot_dopamine_activity(time_points, dopamine_activities, p_values)
    
    # Plot peaks vs probability
    plt.figure(figsize=(8, 6))
    plt.plot(p_values, stimulus_peaks, color=plt.cm.viridis(0.0), label='Stimulus response')
    plt.plot(p_values, reward_peaks, color=plt.cm.viridis(0.8), label='Reward response')
    plt.xlabel('Reward probability (p)')
    plt.ylabel('Peak dopamine response')
    plt.legend()
    plt.title('Dopamine Response vs Reward Probability')
    plt.savefig('results/dopamine_response_vs_probability')
    plt.show()

if __name__ == "__main__":
    exercise_4()
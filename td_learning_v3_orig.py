import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def stimulus_function(t):
    return 1.0 if np.isclose(t, 10.0) else 0.0

def reward_function(t, reward_present=True):
    """Reward function with option to omit reward"""
    if not reward_present:
        return 0.0
    mean = 20.0
    std = 1.0
    return 0.5 * np.exp(-((t - mean) ** 2) / (2 * (std ** 2)))

def get_state(history, feature_detector="box_car"):
    if feature_detector == "box_car":
        box_car_state = np.zeros(len(history))
        for i in range(len(history)):
            box_car_state[i] = sum(history[-(i+1):])
        return box_car_state
    else:
        raise ValueError("Only box_car feature detector supported")

def get_value(state, weights):
    return np.sum(state * weights)

def get_td_error(state, next_state, reward, weights, gamma):
    current_value = get_value(state, weights)
    next_value = get_value(next_state, weights)
    td_error = reward + gamma * next_value - current_value
    return td_error

def model_dopamine(x, alpha=6.0, beta=6.0, x_star=0.27):
    """Model dopamine response with limited dynamic range"""
    if x < 0:
        return x/alpha
    elif x < x_star:
        return x
    else:
        return x_star + (x - x_star)/beta

def run_partial_reinforcement(p=0.5, n_trials=1000, feature_detector="box_car", t_max=25.0, 
                            dt=0.5, memory_span=12.0, epsilon=0.01, gamma=1.0):
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
    
    return time_points, values_history, td_errors_history, reward_present

def analyze_last_100_trials(time_points, values_history, td_errors_history, reward_present):
    """Analyze results from last 100 trials"""
    
    # Get last 100 trials
    last_100_values = values_history[-100:, :]
    last_100_td_errors = td_errors_history[-100:, :]
    last_100_reward = reward_present[-100:]
    
    # Separate rewarded and unrewarded trials
    rewarded_values = last_100_values[last_100_reward]
    unrewarded_values = last_100_values[~last_100_reward]
    rewarded_td_errors = last_100_td_errors[last_100_reward]
    unrewarded_td_errors = last_100_td_errors[~last_100_reward]
    
    # Calculate means
    mean_all_values = np.mean(last_100_values, axis=0)
    mean_rewarded_values = np.mean(rewarded_values, axis=0)
    mean_unrewarded_values = np.mean(unrewarded_values, axis=0)
    
    mean_all_td_errors = np.mean(last_100_td_errors, axis=0)
    mean_rewarded_td_errors = np.mean(rewarded_td_errors, axis=0)
    mean_unrewarded_td_errors = np.mean(unrewarded_td_errors, axis=0)
    
    # Convert TD errors to dopamine signal
    dopamine_signal = np.vectorize(model_dopamine)(mean_all_td_errors)
    
    return {
        'values': (mean_all_values, mean_rewarded_values, mean_unrewarded_values),
        'td_errors': (mean_all_td_errors, mean_rewarded_td_errors, mean_unrewarded_td_errors),
        'dopamine': dopamine_signal
    }

def plot_partial_reinforcement_results(time_points, analysis_results, p):
    """Plot results from partial reinforcement analysis"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot value estimates
    axes[0].plot(time_points, analysis_results['values'][0], 'k-', label='All trials')
    axes[0].plot(time_points, analysis_results['values'][1], 'g-', label='Rewarded')
    axes[0].plot(time_points, analysis_results['values'][2], 'r-', label='Unrewarded')
    axes[0].set_ylabel('Value Estimate')
    axes[0].set_title(f'Value Function (p={p})')
    axes[0].legend()
    
    # Plot TD errors
    axes[1].plot(time_points, analysis_results['td_errors'][0], 'k-', label='All trials')
    axes[1].plot(time_points, analysis_results['td_errors'][1], 'g-', label='Rewarded')
    axes[1].plot(time_points, analysis_results['td_errors'][2], 'r-', label='Unrewarded')
    axes[1].set_ylabel('TD Error')
    axes[1].set_title('TD Error')
    axes[1].legend()
    
    # Plot dopamine signal
    axes[2].plot(time_points, analysis_results['dopamine'], 'b-')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('DA(δ(t))')
    axes[2].set_title('Modeled Dopamine Signal')
    
    plt.tight_layout()
    return fig

def analyze_dopamine_peaks(p_values, n_trials=1000):
    """Analyze dopamine peaks for different reward probabilities"""
    stimulus_peaks = []
    reward_peaks = []
    
    for p in p_values:
        # Run simulation
        time_points, values_history, td_errors_history, reward_present = run_partial_reinforcement(
            p=p, n_trials=n_trials)
        
        # Analyze last 100 trials
        analysis = analyze_last_100_trials(time_points, values_history, td_errors_history, reward_present)
        
        # Find peaks around stimulus (10s) and reward (20s) times
        stimulus_idx = np.abs(time_points - 10.0).argmin()
        reward_idx = np.abs(time_points - 20.0).argmin()
        
        # Get dopamine levels at these times
        stimulus_peaks.append(analysis['dopamine'][stimulus_idx])
        reward_peaks.append(analysis['dopamine'][reward_idx])
    
    return stimulus_peaks, reward_peaks

# Run simulations for different reward probabilities
def exercise_5_to_8():
    # Exercise 5-6: Run and analyze p=0.5 case
    print("\nRunning partial reinforcement simulation (p=0.5)...")
    time_points, values_history, td_errors_history, reward_present = run_partial_reinforcement(p=0.5)
    analysis_results = analyze_last_100_trials(time_points, values_history, td_errors_history, reward_present)
    plot_partial_reinforcement_results(time_points, analysis_results, p=0.5)
    plt.show()
    return 0

    # Exercise 7-8: Analyze multiple reward probabilities
    print("\nAnalyzing different reward probabilities...")
    p_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    stimulus_peaks, reward_peaks = analyze_dopamine_peaks(p_values)
    
    # Plot peaks vs probability
    plt.figure(figsize=(8, 6))
    plt.plot(p_values, stimulus_peaks, 'b-', label='Stimulus response')
    plt.plot(p_values, reward_peaks, 'r-', label='Reward response')
    plt.xlabel('Reward probability (p)')
    plt.ylabel('Peak dopamine response')
    plt.legend()
    plt.title('Dopamine Response vs Reward Probability')
    plt.show()

if __name__ == "__main__":
    exercise_5_to_8()
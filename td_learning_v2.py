"""
Problem with this version is that it has an finite stimulus (from 0s to 25s), so in the calc
of past stimuli in box-calc the accmulations stop once the negative times are reached leading to missing
1.0 in the status matrix. The tapped_line_states is computed correctly
"""


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def generate_stimulus_reward(t: np.ndarray, stim_time: float = 10.0, 
                           reward_time: float = 20.0, reward_std: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate stimulus spike and Gaussian reward signal."""
    # Stimulus spike
    stimulus = np.zeros_like(t)
    stimulus[t == stim_time] = 1.0
    
    # Gaussian reward
    reward = 0.5 * np.exp(-((t - reward_time) ** 2) / (2 * reward_std ** 2))
    
    return stimulus, reward

def tapped_line_states(stimulus: np.ndarray, memory_span: float, dt: float) -> np.ndarray:
    """Compute tapped delay line states for the stimulus."""
    memory_steps = int(memory_span / dt)
    n_time_steps = len(stimulus)
    states = np.zeros((n_time_steps, memory_steps + 1))
    
    for t in range(n_time_steps):
        for tau in range(min(t + 1, memory_steps + 1)):
            if t - tau >= 0:
                states[t, tau] = stimulus[t - tau]
            
    return states

def stimulus_function(t: float) -> float:
    return 1.0 if np.isclose(t, 10.0) else 0.0
    
def boxcar_states(stimulus: np.ndarray, memory_span: float, dt: float) -> np.ndarray:
    """Compute boxcar states for the stimulus."""
    memory_steps = int(memory_span / dt)
    n_time_steps = len(stimulus)
    states = np.zeros((n_time_steps, memory_steps + 1))
    
    # Save state matrix for one time point
    for t in range(n_time_steps):
        #for tau in range(memory_steps):
            
            #for i in range(tau):
                #states[t, tau] += stimulus_function(t-i)
            
        # The inner loop iterates over 𝜏, which determines the length of the memory window.
        for tau in range(min(t + 1, memory_steps + 1)):
            states[t, tau] = sum(stimulus[max(0, t - tau) : t + 1]) 
            
    return states

def td_learning(stimulus: np.ndarray, reward: np.ndarray, memory_span: float, dt: float,
                feature_type: str, gamma: float, epsilon: float, n_trials: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Implement TD learning with function approximation.
    Computes states fresh for each trial.
    Returns weights and value estimates for selected trials.
    """
    # Compute states to get shape to initialize weights
    states = (boxcar_states if feature_type == 'boxcar' else tapped_line_states)(stimulus, memory_span, dt)
    np.savetxt('states.txt', states.astype(int))
    
    weights = np.zeros(states.shape[1])
    
    example_trials = list(range(0, n_trials, 10))  # Every 10th trial (Required 21 example trials plotted)
    value_histories = []
    
    for trial in range(n_trials):
        # Compute states fresh for this trial
        states = (boxcar_states if feature_type == 'boxcar' else tapped_line_states)(stimulus, memory_span, dt)
        
        # Store value estimates for example trials
        if trial in example_trials:
            values = np.array([np.dot(states[t], weights) for t in range(len(states))])
            value_histories.append(values)
        
        # TD learning
        for t in range(len(states) - 1):
            v_t = np.dot(states[t], weights)
            v_next = np.dot(states[t + 1], weights)
            delta = reward[t] + gamma * v_next - v_t
            weights += epsilon * delta * states[t]
    
    return weights, value_histories

def compute_td_variables(values: np.ndarray, reward: np.ndarray, 
                        dt: float, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute temporal difference and TD error for given values."""
    # Temporal difference of value
    delta_v = np.zeros_like(values)
    delta_v[1:] = gamma * values[1:] - values[:-1]
    
    # TD error
    td_error = np.zeros_like(values)
    td_error[1:] = reward[:-1] + delta_v[1:]
    
    return delta_v, td_error

def plot_learning_results(t: np.ndarray, stimulus: np.ndarray, reward: np.ndarray,
                         value_histories: List[np.ndarray], dt: float, gamma: float, feature_type):
    """Plot stimulus, reward, and learning results."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    
    # Plot stimulus and reward
    axes[0].plot(t, stimulus, label='Stimulus', color=plt.cm.winter(0.0), linewidth=2)
    axes[0].plot(t, reward, label='Reward', color=plt.cm.winter(1.0), linewidth=2)
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Stimulus and Reward')
    axes[0].legend(loc='upper right')
    
    # Create color map and legend entries - reversed gradient
    colors = plt.cm.winter(np.linspace(1, 0, len(value_histories)))  # Reversed order winter (viridis also possible)
    legend_elements = []
    
    # Plot value estimates
    for i, values in enumerate(value_histories):
        line = axes[1].plot(t, values, color=colors[i], alpha=0.7, label=f'Trial {i*10 + 1}', linewidth=1.5)
        
        if i in [0, len(value_histories)//2, len(value_histories)-1]:  # Add only some trials to legend
            legend_elements.append(line[0])
            
    axes[1].set_ylabel('Value Estimate')
    axes[1].set_title('Value Function Over Learning')
    axes[1].legend(handles=legend_elements, labels=[f'Trial {i*10 + 1}' for i in [0, len(value_histories)//2, len(value_histories)-1]], loc='upper right')
    
    # Plot delta value function
    for i, values in enumerate(value_histories):
        delta_v, _ = compute_td_variables(values, reward, dt, gamma)
        axes[2].plot(t, delta_v, color=colors[i], alpha=0.7, linewidth=1.5)
        
        if i in [0, len(value_histories)//2, len(value_histories)-1]:  # Add only some trials to legend
            legend_elements.append(line[0])
            
    axes[2].set_ylabel('ΔV(t)')
    axes[2].set_title('Temporal Difference of Value Function')
    
    # Plot TD error
    for i, values in enumerate(value_histories):
        _, td_error = compute_td_variables(values, reward, dt, gamma)
        axes[3].plot(t, td_error, color=colors[i], alpha=0.7, linewidth=1.5)

            
    axes[3].set_ylabel('TD Error δ(t)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('TD Error Over Learning')
    
    plt.tight_layout()
    plt.savefig(f'results//ex{4 if feature_type == 'boxcar' else 3}_td_learning_{feature_type}.png')
    plt.show()

def run_simulation(feature_type='boxcar'):
    """Run TD learning simulation with specified feature type."""
    # Main simulation parameters
    t = np.arange(0, 25.1, 0.5)  # Time points
    dt = 0.5  # Time step
    gamma = 1.0  # No discounting
    memory_span = 12.0  # Memory span in seconds
    n_trials = 201  # Number of trials

    # Set learning rate based on feature type
    epsilon = 0.01 if feature_type == 'boxcar' else 0.2

    # Generate stimulus and reward
    stimulus, reward = generate_stimulus_reward(t)

    # Run TD learning with feature computation for each trial
    weights, value_histories = td_learning(stimulus, reward, memory_span, dt, feature_type, gamma, epsilon, n_trials)

    # Plot results
    plot_learning_results(t, stimulus, reward, value_histories, dt, gamma, feature_type)
    plt.suptitle(f'TD Learning with {feature_type.title()} states', y=1.02, fontsize=14)
    
def main():
    # Run simulation with boxcar states
    print("Running simulation with boxcar states...")
    run_simulation('boxcar')
    
    # Run simulation with tapped delay line states
    print("Running simulation with tapped delay line states...")
    run_simulation('tapped')

if __name__ == "__main__":
    main()
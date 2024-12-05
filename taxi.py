import gymnasium as gym
import numpy as np
import random

env = gym.make("Taxi-v3", render_mode='ansi')
state, info = env.reset()
print(env.render())



# Initialize environment
env = gym.make("Taxi-v3")
num_states = env.observation_space.n
num_actions = env.action_space.n

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.999  # Decay rate for exploration
min_epsilon = 0.1  # Minimum exploration rate
num_episodes = 10000 # Num of training episodes

# Initialize Q-table
Q = np.zeros((num_states, num_actions))

# Training the Q-learning agent
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    actions_taken = 0

    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit

        # Take the action
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        actions_taken += 1

        # Update Q-value using the Bellman equation
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        # Transition to the next state
        state = next_state

    # Decay the exploration rate
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Optional: Print the total reward per episode
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# Average performance evaluation
total_rewards = []
total_actions = []

for _ in range(10):  # Run the evaluation 10 times
    state, info = env.reset()
    done = False
    total_reward = 0
    actions_taken = 0

    while not done:
        action = np.argmax(Q[state])  # Always exploit
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        actions_taken += 1
        state = next_state

    total_rewards.append(total_reward)
    total_actions.append(actions_taken)

average_reward = np.mean(total_rewards)
average_actions = np.mean(total_actions)

print(f"Average Total Reward over 10 runs: {average_reward}")
print(f"Average Number of Actions over 10 runs: {average_actions}")

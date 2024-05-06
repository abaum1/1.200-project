import numpy as np


def test_q_learning_model_with_q_table(q_table, environment, num_episodes=100):
    total_rewards = []
    assigned_route_1 = []
    assigned_route_2 = []
    assigned_route_3 = []
    assigned_route_4 = []

    for episode in range(num_episodes):
        state = environment.reset()
        total_reward = 0
        done = False

        while not done:
            action = np.argmax(
                q_table[state])  # Greedy action selection using Q-table because our goal is to assess 
            # the performance of the learned policy. Can revisit this later.
            next_state, reward, done, _ = environment.step(action)
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        assigned_route_1.append(action[0])
        assigned_route_2.append(action[1])
        assigned_route_3.append(action[2])
        assigned_route_4.append(action[3])

    average_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")

    return total_rewards, average_reward, assigned_route_1, assigned_route_2, assigned_route_3, assigned_route_4


# Example usage:f
# Save Q-table
# save_q_table(q_table, 'q_table.pkl')

# Load Q-table
# q_table = load_q_table('q_table.pkl')

# Test Q-learning model with the loaded Q-table
# total_rewards = test_q_learning_model_with_q_table(q_table, environment, num_episodes=100)

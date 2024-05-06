import numpy as np
import settings
from q_learning.helpers import generate_state, load_q_table, filter_for_valid_actions
from typing import List, Tuple
from q_learning.environment import Environment
import pandas as pd


def run_evaluation(all_states: List[Tuple[int]], all_actions: List[Tuple[int]],
                   discount_factor: float, learning_rt: float,
                   explore_rt: float) -> Tuple[any]:

    q_table = load_q_table(
        f'src/results/q_table_s1_steps{settings.LEARNING_STEPS}_h{settings.TIME_HORIZON_HOURS}_expl{settings.EXPLORATION_RATE}_dis{settings.DISCOUNT_FACTOR}_learn{settings.LEARNING_RATE}_high{settings.PERFORMANCE_PENALTY["high"]}_low{settings.PERFORMANCE_PENALTY["low"]}.pkl'
    )

    environment = Environment(len(all_states), len(all_actions), learning_rt,
                              discount_factor, explore_rt)

    total_rewards, average_reward, assigned_route_results = test_q_learning_model_with_q_table(
        all_states,
        all_actions,
        q_table,
        environment,
        num_episodes=settings.TESTING_NUM_EPISODES)

    return total_rewards, average_reward, assigned_route_results


def test_q_learning_model_with_q_table(all_states,
                                       all_actions,
                                       q_table,
                                       environment,
                                       num_episodes=100):
    total_rewards = []
    assigned_route_1 = []
    assigned_route_2 = []
    assigned_route_3 = []
    assigned_route_4 = []

    # initial state
    state = generate_state(0, settings.TIME_PERIOD_LENGTH,
                           settings.ROUTE_HEADWAYS,
                           settings.DAILY_TOTAL_EXTRABOARD)

    for episode in range(num_episodes):
        total_reward = 0
        done = False

        while not done:
            state_idx = all_states.index(state) #TODO: how are we able to get to states that start with 6?

            possible_action_indices = filter_for_valid_actions(
                all_actions,
                state)  # which actions can you take from a given state

            action_idx, action = environment.choose_action( # TODO: use the given  table
                state_idx, possible_action_indices, all_actions, True, q_table)  # Greedy action selection using Q-table because our goal is to assess
            # the performance of the learned policy. Can revisit this later.
            next_state, reward, done = environment.step(state, action)
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        assigned_route_1.append(
            action[0]
        )  # for each episode append the number of operators assinged to each route so can plot episode vs. number assigned for each route
        assigned_route_2.append(action[1])
        assigned_route_3.append(action[2])
        assigned_route_4.append(action[3])

    assigned_route_results = pd.DataFrame({'episode': range(num_episodes)})
    assigned_route_results["route_1"] = assigned_route_1
    assigned_route_results["route_2"] = assigned_route_2
    assigned_route_results["route_3"] = assigned_route_3
    assigned_route_results["route_4"] = assigned_route_4

    average_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")

    return total_rewards, average_reward, assigned_route_results


# Example usage:f
# Save Q-table
# save_q_table(q_table, 'q_table.pkl')

# Load Q-table
# q_table = load_q_table('q_table.pkl')

# Test Q-learning model with the loaded Q-table
# total_rewards = test_q_learning_model_with_q_table(q_table, environment, num_episodes=100)

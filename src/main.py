# from 1_200_project import settings
from q_learning.environment import Environment
from q_learning.helpers import generate_state, filter_for_valid_actions
from itertools import product
import settings
import state_probabilities
import numpy as np
from collections import deque
import pandas as pd
from typing import List, Tuple


def run_model(all_states: List[Tuple[int]], all_actions: List[Tuple[int]],
              environment) -> pd.DataFrame:

    last_100_rewards = deque(maxlen=100)
    results_by_episode = []

    for episode in range(settings.LEARNING_STEPS):
        state = generate_state(0, settings.TIME_PERIOD_LENGTH,
                               settings.ROUTE_HEADWAYS,
                               settings.DAILY_TOTAL_EXTRABOARD)
        done = False

        while not done:
            state_idx = all_states.index(state)
            possible_action_indices = filter_for_valid_actions(
                all_actions,
                state)  # which actions can you take from a given state
            action_idx, action = environment.choose_action(
                state_idx, possible_action_indices,
                all_actions)  # filter for valid actions only
            next_state, reward, done = environment.step(state, action)
            last_100_rewards.append(reward)
            next_state_idx = all_states.index(
                next_state) if next_state[0] < 6 else None
            environment.update_q_table(state_idx, action_idx, reward,
                                       next_state_idx, done)
            state = next_state

        rolling_average_reward = sum(last_100_rewards) / len(last_100_rewards)
        results_by_episode.append({
            'episode': episode,
            'average_q_value': None,
            'rolling_avg_reward': rolling_average_reward
        })
    results_by_episode_df = pd.DataFrame(results_by_episode)
    return results_by_episode_df


if __name__ == "__main__":
    print('hello')

    #### environment setup

    all_states = state_probabilities.generate_all_states()
    all_actions = list(
        product(settings.ASSIGNMENT_OPTIONS_PER_ROUTE,
                settings.ASSIGNMENT_OPTIONS_PER_ROUTE,
                settings.ASSIGNMENT_OPTIONS_PER_ROUTE,
                settings.ASSIGNMENT_OPTIONS_PER_ROUTE))

    # hyperparameter tuning
    discount_factors = [0.9, 0.95, 0.99]
    learning_rates = [0.1, 0.05, 0.01, 0.005]
    exploration_rates = [.01, .05, .1, .12]

    for discount, learning_rt, explore_rt in product(discount_factors,
                                                     learning_rates,
                                                     exploration_rates):
        print(
            f"Running model with: discount_factor={discount}, learning_rate={learning_rt}, exploration_rate={explore_rt}"
        )

        environment = Environment(len(all_states), len(all_actions),
                                  learning_rt, discount, explore_rt)

        results_by_episode = run_model(all_states, all_actions, environment)
        results_by_episode.to_csv(
            f'src/results/results_s1_steps{settings.LEARNING_STEPS}_horizon{settings.TIME_HORIZON_HOURS}_expl{explore_rt}_dis{discount}_learning{learning_rt}_0basereward.csv'
        )

    # if (episode + 1) % 100 == 0:

    #     print(
    #         f"Episode: {episode:4d} | Nonzero Q-table elements: {np.count_nonzero(environment.q_table)} | Rolling Average Reward: {rolling_average_reward:.2f}"
    #     )
    # print(environment.q_table)

    # track the mean of the latest 100 rewards (rolling average)
    # sum/average the reward for each step per episode
    # x axis: episide number, y is mean reward
    # rolling average of each 50 episides or just plot every 50

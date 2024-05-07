# from 1_200_project import settings
from q_learning.environment import Environment
import q_learning.helpers as helpers
from itertools import product
import settings
from collections import deque
import pandas as pd
from typing import List, Tuple
from q_learning.testing import run_evaluation


def run_model(all_states: List[Tuple[int]], all_actions: List[Tuple[int]],
              discount_factor, learning_rt, explore_rt):

    environment = Environment(len(all_states), len(all_actions), learning_rt,
                              discount_factor, explore_rt)

    last_100_rewards = deque(maxlen=100)
    results_by_episode = []

    for episode in range(settings.LEARNING_STEPS):
        state = helpers.generate_state(0, settings.TIME_PERIOD_LENGTH,
                                       settings.ROUTE_HEADWAYS,
                                       settings.DAILY_TOTAL_EXTRABOARD)
        done = False

        while not done:
            state_idx = all_states.index(state)
            possible_action_indices = helpers.filter_for_valid_actions(
                all_actions,
                state)  # which actions can you take from a given state
            action_idx, action = environment.choose_action(
                state_idx, possible_action_indices, all_actions)
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
    final_q_table = environment.q_table
    return results_by_episode_df, final_q_table


def run_with_grid_search(all_states: List[Tuple[int]],
                         all_actions: List[Tuple[int]],
                         discount_factors: List[float],
                         learning_rates: List[float],
                         exploration_rates: List[float]):
    for discount_factor, learning_rt, explore_rt in product(
            discount_factors, learning_rates, exploration_rates):
        print(
            f"Running model with: discount_factor={discount_factor}, learning_rate={learning_rt}, exploration_rate={explore_rt}"
        )

        results_by_episode, final_q_table = run_model(all_states, all_actions,
                                                      discount_factor,
                                                      learning_rt, explore_rt)
        results_by_episode.to_csv(
            f'src/results/results_s1_steps{settings.LEARNING_STEPS}_h{settings.TIME_HORIZON_HOURS}_expl{explore_rt}_dis{discount_factor}_learn{learning_rt}_high{settings.PERFORMANCE_PENALTY["high"]}_low{settings.PERFORMANCE_PENALTY["low"]}.csv'
        )

    pd.DataFrame(final_q_table).to_csv(
        f'src/results/q_table_s1_steps{settings.LEARNING_STEPS}_h{settings.TIME_HORIZON_HOURS}_expl{explore_rt}_dis{discount_factor}_learn{learning_rt}_high{settings.PERFORMANCE_PENALTY["high"]}_low{settings.PERFORMANCE_PENALTY["low"]}.csv'
    )


def run_with_constants(all_states: List[Tuple[int]],
                       all_actions: List[Tuple[int]], discount_factor: float,
                       learning_rt: float, explore_rt: float):
    print("about to run model")
    results_by_episode, final_q_table = run_model(all_states, all_actions,
                                                  discount_factor, learning_rt,
                                                  explore_rt)
    results_by_episode.to_csv(
        f'src/results/results_s1_steps{settings.LEARNING_STEPS}_h{settings.TIME_HORIZON_HOURS}_expl{explore_rt}_dis{discount_factor}_learn{learning_rt}_high{settings.PERFORMANCE_PENALTY["high"]}_low{settings.PERFORMANCE_PENALTY["low"]}.csv'
    )

    helpers.save_q_table(
        final_q_table,
        f'src/results/q_table_s1_steps{settings.LEARNING_STEPS}_h{settings.TIME_HORIZON_HOURS}_expl{explore_rt}_dis{discount_factor}_learn{learning_rt}_high{settings.PERFORMANCE_PENALTY["high"]}_low{settings.PERFORMANCE_PENALTY["low"]}.pkl'
    )


if __name__ == "__main__":

    all_states = helpers.generate_all_states()
    all_actions = helpers.generate_potential_actions(
        settings.ASSIGNMENT_OPTIONS_PER_ROUTE
    )  # set to ASSIGNMENT_OPTIONS_PER_ROUTE for RL policy,
    # set to DAILY_TOTAL_EXTRABOARD for naive policy because the naive policy requires a larger action space

    # run_with_grid_search(all_states, all_actions,
    #                      settings.GRID_SEARCH_DISCOUNT_FACTORS,
    #                      settings.GRID_SEARCH_LEARNING_RATES,
    #                      settings.GRID_SEARCH_EXPLORATION_RATES)

    # run_with_constants(all_states, all_actions, settings.DISCOUNT_FACTOR,
    #                    settings.LEARNING_RATE, settings.EXPLORATION_RATE)

    assigned_route_results = run_evaluation(all_states, all_actions,
                                            settings.DISCOUNT_FACTOR,
                                            settings.LEARNING_RATE,
                                            settings.EXPLORATION_RATE, 'RL')


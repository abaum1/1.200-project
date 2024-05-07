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

    assigned_route_results = test_q_learning_model_with_q_table(
        all_states,
        all_actions,
        q_table,
        environment,
        num_episodes=settings.TESTING_NUM_EPISODES)
    assigned_route_results.to_csv(f'src/results/validation/testing_s1_steps{settings.LEARNING_STEPS}_h{settings.TIME_HORIZON_HOURS}_expl{settings.EXPLORATION_RATE}_dis{settings.DISCOUNT_FACTOR}_learn{settings.LEARNING_RATE}_high{settings.PERFORMANCE_PENALTY["high"]}_low{settings.PERFORMANCE_PENALTY["low"]}.csv')
    return assigned_route_results


def test_q_learning_model_with_q_table(all_states,
                                       all_actions,
                                       q_table,
                                       environment,
                                       num_episodes=100):
    assigned_route_1 = []
    assigned_route_2 = []
    assigned_route_3 = []
    assigned_route_4 = []
    num_unique_actions = []
    mean_rewards = []

    for episode in range(num_episodes):
        # initial state
        state = generate_state(0, settings.TIME_PERIOD_LENGTH,
                               settings.ROUTE_HEADWAYS,
                               settings.DAILY_TOTAL_EXTRABOARD)
        done = False

        actions_for_episode = []
        rewards_for_episode = []

        for time_period in range(6):
            state_idx = all_states.index(
                state
            )  # TODO: how are we able to get to states that start with 6?

            possible_action_indices = filter_for_valid_actions(
                all_actions,
                state)  # which actions can you take from a given state

            action_idx, action = environment.choose_action(
                state_idx, possible_action_indices, all_actions, True, q_table
            )  # Greedy action selection using Q-table because our goal is to assess
            # the performance of the learned policy. Can revisit this later.
            next_state, reward, done = environment.step(state, action)
            #TODO: need to break if done is true becAuse we are not calling update_q_table so it is not handled
            rewards_for_episode.append(reward)
            state = next_state
            actions_for_episode.append(
                action
            )  # get the list of action tuples that were taken during that episode

        assigned_route_1.append(
            sum([action[0] for action in actions_for_episode]))
        assigned_route_2.append(
            sum([action[1] for action in actions_for_episode])
        )  # sum the number of operators assigned to each route for all the actions that were taken during the episode
        assigned_route_3.append(
            sum([action[2] for action in actions_for_episode]))
        assigned_route_4.append(
            sum([action[3] for action in actions_for_episode]))

        num_unique_actions.append(len(set(actions_for_episode)))
        mean_rewards.append(np.mean(rewards_for_episode))

    assigned_route_results = pd.DataFrame({'episode': range(num_episodes)})
    assigned_route_results["route_1"] = assigned_route_1
    assigned_route_results["route_2"] = assigned_route_2
    assigned_route_results["route_3"] = assigned_route_3
    assigned_route_results["route_4"] = assigned_route_4
    assigned_route_results["num_unique_actions"] = num_unique_actions
    assigned_route_results["mean_reward"] = mean_rewards

    return assigned_route_results


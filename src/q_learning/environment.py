import numpy as np
import settings as settings
import q_learning.helpers as helpers
from typing import List, Any, Tuple


class Environment:

    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate=settings.LEARNING_RATE,
                 discount_factor=settings.DISCOUNT_FACTOR,
                 exploration_rate=settings.EXPLORATION_RATE):
        self.q_table = np.zeros((num_states, num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate\


    def choose_naive_action(
            self, state: Tuple[int], possible_action_indices: List[int],
            all_actions: List[Tuple[int]]) -> Tuple[int, Tuple[int, int]]:
        '''This implements the naive policy where extraboard is assigned in a greedy manner in order of
        routes with most missing trips to least.'''

        remaining_extraboard = state[-1]

        sorted_route_indices = sorted(list(enumerate(state[1:5])),
                                      key=lambda x: x[1],
                                      reverse=True)
        indices_increasing_order = [index for index, _ in sorted_route_indices]

        action = np.zeros(4)
        for index in indices_increasing_order:
            action[index] = min(
                remaining_extraboard, state[index + 1]
            )  # action is set to the number of missing trips that are assigned to that route
            remaining_extraboard = remaining_extraboard - action[index]

        assert all_actions.index(action) in possible_action_indices

        return action

    def choose_action(self,
                      state_idx: int,
                      possible_action_indices: List[int],
                      all_actions: List[Tuple[int]],
                      force_greedy=False,
                      q_table=None) -> Tuple[int, Tuple[int, int]]:
        '''This implements the epsilon greedy strategy where epsilon is randomly generated and
        if it is less than the exploration rate, we "explore" by choosing a random action,
        otherwise we exploit by choosing the action that has the highest reward. When we use this fn
        for testing, we will use a pre-computed q table and optionally force the function to always
        use the greedy policy regardless of the randomly generated epsilon.'''

        selected_q_table = q_table if q_table is not None else self.q_table

        epsilon = np.random.uniform(0, 1)
        if epsilon < self.exploration_rate and force_greedy is False:
            action_idx = np.random.choice(
                possible_action_indices
            )  # randomly choose an index that corresponds to the index in the action array
        else:
            # choose the action_idx with the highest q value
            possible_action_idx = np.argmax(
                selected_q_table[state_idx][possible_action_indices])
            action_idx = possible_action_indices[possible_action_idx]

        assert action_idx in possible_action_indices

        return action_idx, all_actions[action_idx]

    def update_q_table(self,
                       state_idx: int,
                       action_idx: int,
                       reward: float,
                       next_state_idx: int = None,
                       done: bool = False) -> None:
        if done:
            q_value = self.q_table[
                state_idx,
                action_idx]  # get current q value for the state and action index
            new_q_value = q_value + self.learning_rate * (reward - q_value)
            self.q_table[state_idx, action_idx] = new_q_value
        else:
            q_value = self.q_table[
                state_idx,
                action_idx]  # get current q value for the state and action index
            max_next_q_value = np.max(
                self.q_table[next_state_idx]
            )  # get the best action of the ones currently saved in the q table for this state
            new_q_value = q_value + self.learning_rate * (
                reward + self.discount_factor * max_next_q_value - q_value
            )  # q update equation to determine new q value
            self.q_table[state_idx, action_idx] = new_q_value

    def get_reward(self, state: Tuple[int], action: Tuple[int]) -> float:
        base_reward = 0
        missing_low_vulnerability = (state[0] + state[2]) - (action[0] +
                                                             action[2])
        missing_high_vulnerability = (state[1] + state[3]) - (action[1] +
                                                              action[3])
        # the higher the combined penalty, the worse the reward is.
        return base_reward - (
            settings.PERFORMANCE_PENALTY['high'] * missing_high_vulnerability +
            settings.PERFORMANCE_PENALTY['low'] * missing_low_vulnerability)

    # think about demand for each route (passengers impacted). or is this encapsuated in the

    def transition(self, state: tuple[int], action: tuple[int]) -> tuple[int]:
        '''Given the current state and action, compute what the next state will be. Currently this is influenced by the action
        directly because of the number of available extraboard is reduced, but it is partially stochastic because the number of missing
        trips at each time period is randomly generated.'''

        remaining_extraboard_after_action = state[-1] - sum(action)
        next_state = helpers.generate_state(state[0] + 1,
                                            settings.TIME_PERIOD_LENGTH,
                                            settings.ROUTE_HEADWAYS,
                                            remaining_extraboard_after_action)
        return next_state

    def step(self, state: Tuple[int], action: Tuple[int]) -> List[Any]:
        reward = self.get_reward(state, action)
        next_state = self.transition(state, action)
        done = True if next_state[
            0] == 6 else False  # each episode is a day. There are 6 timesteps within a day. When the day finishes, the episode is done.
        return next_state, reward, done

    def reset(self) -> None:
        self.q_table = np.zeros_like(self.q_table)

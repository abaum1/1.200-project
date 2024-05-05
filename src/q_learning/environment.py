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
        self.q_table = np.zeros(
            (num_states,
             num_actions))  # TODO: do I need to initialize this differently?
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state_idx: int,
                      possible_actions: List[Tuple[int]]) -> tuple[int]:
        '''This implements the epsilon greedy strategy where epsilon is randomly generated and if it is less than the exploration
        rate, we "explore" by choosing a random action, otherwise we explot by choosing the action that has the highest reward.'''
        epsilon = np.random.uniform(0, 1)
        #TODO: this is retunring 0 because the q table is empty.
        if epsilon < self.exploration_rate:
            return np.random.choice(len(self.q_table[state_idx])) # randomly choose an index
        else:
            return np.argmax(
                self.q_table[state_idx]
            )  # choose the action_idx with the highest q value
        # TODO: need to filter only the valid actions which are a fn of extraboard. action idx.
        # if all or some are 0 (of it the max is 0), if theres a tie, randomly choose. the actions are not 0, the values are 0.

    #TODO:
    # filter out invalid actions and change data strucure so that we actually return a tuple
    # consider making the reward negative

    def update_q_table(self, state_idx: int, action_idx: int, reward: float,
                       next_state_idx: int, done: bool) -> None:
        if done:
            # TODO: if it's a terminal state then there's no next_q_value
            NotImplementedError
        else:
            q_value = self.q_table[
                state_idx,
                action_idx]  # get current q value for the state and action index
            max_next_q_value = np.max(
                self.q_table[next_state_idx]
            )  # get the best action of the ones currently saved in the q table for this state
            new_q_value = q_value + self.learning_rate * (
                reward + self.discount_factor * max_next_q_value -
                q_value  #TODO: because the reward is positive Q values will be biased higher. 
            )  # q update equation to determine new q value
            self.q_table[state_idx, action_idx] = new_q_value
            print("q table updated")

    def get_reward(state: tuple[int], action: tuple[int]) -> float:
        # we use the route type because this is not encoded in the state
        # we want the agent to "learn" which routes are more vulnerable than others
        base_reward = 1
        missing_low_perf = (state[0] + state[2]) - (
            action[0] + action[2]
        )  # the current number of missing trips on the low performance routes -
        # the trips that have been filled on those routes by the specific action
        # demand will be correlated with frequency. maybe keepit simple if the performance penality is assumed to encapsulate all those
        # can also vary the weights as part of the scenario analysis
        missing_high_perf = (state[1] + state[3]) - (action[1] + action[3])
        return base_reward - (
            settings.PERFORMANCE_PENALTY['high'] * missing_high_perf +
            settings.PERFORMANCE_PENALTY['low'] * missing_low_perf)

    #think about demand for each route (passengers impacted). or is this encapsuated in the

    def transition(state: tuple[int], action: tuple[int]) -> tuple[int]:
        '''Given the current state and action, compute what the next state will be. Currently this is influenced by the action
        directly because of the number of available extraboard is reduced, but it is partially stochastic because the number of missing
        trips at each time period is randomly generated.'''

        remaining_extraboard_after_action = state[-1] - sum(action)
        next_state = helpers.generate_state(state[0] + 1,
                                            settings.TIME_PERIOD_LENGTH,
                                            settings.ROUTE_HEADWAYS,
                                            remaining_extraboard_after_action)
        return next_state

    def step(self, state: tuple[int], action: tuple[int]) -> List[Any]:
        #TODO: this isn't working because action is 0, it should be a tuple. need to fix the update_q_table fn
        reward = self.get_reward(state, action)
        next_state = self.transition(state, action)
        done = True if next_state[
            0] == 480 else False  # each episode is a day. There are 6 timesteps within a day. When the day finishes, the episode is done.
        return next_state, reward, done

    def reset(self) -> None:
        self.q_table = np.zeros_like(self.q_table)

import numpy as np
from scenario1_setup import generate_state
import env as env
import helpers as helpers
# DISCOUNT_FACTOR, ROUTE_HEADWAYS, LEARNING_STEPS, LEARNING_RATE, EXPLORATION_RATE, PERFORMANCE_PENALTY


class Environment:

    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate=env.LEARNING_RATE,
                 discount_factor=env.DISCOUNT_FACTOR,
                 exploration_rate=env.EXPLORATION_RATE):
        self.q_table = np.zeros((num_states, num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        '''This implements the epsilon greedy strategy where epsilon is randomly generated and if it is less than the exploration
        rate, we "explore" by choosing a random action, otherwise we explot by choosing the action that has the highest reward.'''
        epsilon = np.random.uniform(0, 1)
        if epsilon < self.exploration_rate:
            return np.random.choice(len(self.q_table[state]))
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward,
                       next_state):  # this is the step function
        q_value = self.q_table[
            state, action]  # get current q value for the state and action
        max_next_q_value = np.max(
            self.q_table[next_state]
        )  # get the best action of the ones currently saved in the q table for this state
        new_q_value = q_value + self.learning_rate * (
            reward + self.discount_factor * max_next_q_value - q_value
        )  # q update equation to determine new q value
        self.q_table[state, action] = new_q_value

    def get_reward(state, action):
        # we use the route type because this is not encoded in the state
        # we want the agent to "learn" which routes are more vulnerable than others
        base_reward = 1
        missing_low_perf = (state[0] + state[2]) - (
            action[0] + action[2]
        )  # the current number of missing trips on the low performance routes -
        # the trips that have been filled on those routes by the specific action
        missing_high_perf = (state[1] + state[3]) - (action[1] + action[3])
        return base_reward - (
            env.PERFORMANCE_PENALTY['high'] * missing_high_perf +
            env.PERFORMANCE_PENALTY['low'] * missing_low_perf)

    def transition(state, action):
        '''Given the current state and action, compute what the next state will be. Currently this is influenced by the action
        directly because of the number of available extraboard is reduced, but it is partially stochastic because the number of missing
        trips at each time period is randomly generated.'''

        remaining_extraboard_after_action = state[-1] - sum(action)
        next_state = helpers.generate_state(state[0] + 1,
                                            env.TIME_PERIOD_LENGTH,
                                            env.ROUTE_HEADWAYS,
                                            remaining_extraboard_after_action)
        return next_state

    def step(self, state, action):
        reward = self.get_reward(state, action)
        next_state = self.transition(state, action)
        done = True if next_state[
            0] == 480 else False  #TODO: check if this is the right termination logic
        return next_state, reward, done

    def reset(self):
        self.q_table = np.zeros_like(self.q_table)


# Example usage
num_states = 480
num_actions = 81
num_episodes = 30000

environment = Environment(num_states, num_actions)

for episode in range(num_episodes):
    state = generate_state(0, 4, ROUTE_HEADWAYS, 30)
    done = False

    while not done:

        action = environment.choose_action(state)
        next_state, reward, done = environment.step(state, action)
        environment.update_q_table(state, action, reward, next_state)
        state = next_state

    environment.reset()

# from 1_200_project import settings
from q_learning.environment import Environment
from q_learning.helpers import generate_state, filter_for_valid_actions
from itertools import product
import settings
import state_probabilities

if __name__ == "__main__":

    all_states = state_probabilities.generate_all_states()
    all_actions = list(
        product(settings.ASSIGNMENT_OPTIONS_PER_ROUTE,
                settings.ASSIGNMENT_OPTIONS_PER_ROUTE,
                settings.ASSIGNMENT_OPTIONS_PER_ROUTE,
                settings.ASSIGNMENT_OPTIONS_PER_ROUTE))
    print('num states', len(all_states), all_states[:100])
    print('num actions', len(all_actions), all_actions)

    environment = Environment(len(all_states), len(all_actions),
                              settings.LEARNING_RATE, settings.DISCOUNT_FACTOR,
                              settings.EXPLORATION_RATE)

    for episode in range(settings.LEARNING_STEPS):
        state = generate_state(0, settings.TIME_PERIOD_LENGTH,
                               settings.ROUTE_HEADWAYS,
                               settings.DAILY_TOTAL_EXTRABOARD)
        done = False

        while not done:
            state_idx = all_states.index(state)
            possible_actions = filter_for_valid_actions(all_actions, state) # which actions can you take from a given state
            action = environment.choose_action(
                state_idx, possible_actions)  # filter for valid actions only
            action_idx = all_actions.index(action)
            next_state, reward, done = environment.step(state, action)
            next_state_idx = all_states.index(next_state)
            environment.update_q_table(state_idx, action_idx, reward,
                                       next_state_idx, done)
            state = next_state

        environment.reset()

# from 1_200_project import settings
from q_learning.environment import Environment
from q_learning.helpers import generate_state
import sys
import settings
import state_probabilities
# from 1_200_project import settings

# scenarios:
#

if __name__ == "__main__":
    print('sys path:', sys.path)
    print('settings.NUM_STATES', settings.NUM_STATES)

    all_states = state_probabilities.generate_all_states()
    
    environment = Environment(settings.NUM_STATES, settings.NUM_ACTIONS,
                              settings.LEARNING_RATE, settings.DISCOUNT_FACTOR,
                              settings.EXPLORATION_RATE)

    for episode in range(settings.LEARNING_STEPS):
        state = generate_state(0, settings.TIME_PERIOD_LENGTH,
                               settings.ROUTE_HEADWAYS, 30)
        done = False

        while not done:
            state_idx = all_states.index(state)
            action = environment.choose_action(state_idx)
            next_state, reward, done = environment.step(state, action)
            environment.update_q_table(state, action, reward, next_state, done)
            state = next_state

        environment.reset()

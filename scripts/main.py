# Example usage
import settings as settings
from q_learning.environment import Environment
from q_learning.helpers import generate_state

if __name__ == "__main__":
    environment = Environment(settings.NUM_STATES, settings.NUM_ACTIONS)

    for episode in range(settings.NUM_EPISODES):
        state = generate_state(0, settings.TIME_PERIOD_LENGTH,
                               settings.ROUTE_HEADWAYS, 30)
        done = False

        while not done:

            action = environment.choose_action(state)
            next_state, reward, done = environment.step(state, action)
            environment.update_q_table(state, action, reward, next_state)
            state = next_state

        environment.reset()

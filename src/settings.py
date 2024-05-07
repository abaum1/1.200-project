ROUTE_IDS = ['A', 'B', 'C', 'D']
# Route characteristics:
# A low vulnerability, high freq
# B high vulnerability, high freq
# C low vulnerability, low freq
# D high vulnerability, low freq
ROUTE_VULNERABILITY_LEVELS = {'A': 'low', 'B': 'high', 'C': 'low', 'D': 'high'}
ROUTE_FREQUENCY = {'A': 'high', 'B': 'high', 'C': 'low', 'D': 'low'}

# average scheduled headway for routes in each group. Groups are defined as:
# high frequency: scheduled headways < 15 mins
# low frequency: scheduled headways >= 15 mins
# for now start with narrower distribution. May also want to try this with more extreme differences (extremely high/extremely low)
HEADWAYS = {'high': 12, 'low': 18}

# penalty will be used for the reward by subtracting the reward by:
# penaly * n_missing_trips
# This penalizes performance based on staffing differentially by the route frequency. This is linear but the penalty should probably
# be exponential.

PERFORMANCE_PENALTY = {  # high and low refer to vulnerability, in real world inversely related to freq.
    'low': 0.09,  # low vulnerability penality - <15 mins
    'high': 0.14  # high vulnerability penality - >= 15 mins
}

# From Haris: vulnerability becomes a factor if you have different levels of it, more continuous not an attribute.
# can design a study in terms of levels and factors given the computational effort. Look this up. See if there
# is on the importance of the missed trip relative to another. Quinyi thesis looked at this problem for MBTA
# (lit review) masters thesis.
ROUTE_HEADWAYS = [
    HEADWAYS['high'], HEADWAYS['high'], HEADWAYS['low'], HEADWAYS['low']
]

# time period: 1 day (24 hours, 6 4-hour periods).
TIME_PERIOD_HOURS = 4
TIME_HORIZON_HOURS = 24

DAILY_TOTAL_EXTRABOARD = 15  # maybe run it with this and with a higher number closer to reality like 30-40
MAX_MISSING_TRIPS_PCT = 0.3  # I think this is reasonable. A lot of the routes seem to be missing 7-15% so uniform dist over [0, .3] seems ok.

# best combination from grid search: discount: 0.95, learning: 0.01 exploration: 0.1
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 0.1
LEARNING_RATE = 0.01  # how quickly you want to update the Q table. If too small, too slow. If too fast, the update function is too jerky.
LEARNING_STEPS = 10000  # 6 timesteps per episode, 30k episodes = days simulating.
TESTING_NUM_EPISODES = 20

GRID_SEARCH_DISCOUNT_FACTORS = [0.9, 0.95, 0.99]
GRID_SEARCH_LEARNING_RATES = [0.1, 0.05, 0.01, 0.005]
GRID_SEARCH_EXPLORATION_RATES = [.01, .05, .1, .12]

NUM_STATES = 480
NUM_ACTIONS = 81  # 3*3*3*3

ASSIGNMENT_OPTIONS_PER_ROUTE = range(
    0, 3
)  # TODO: maybe alter this if I don't want to restrict actions to (0,1, 2)
TIME_PERIOD_LENGTH = 4

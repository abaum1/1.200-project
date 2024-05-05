import numpy as np
import pandas as pd
from typing import List, Tuple

#
# H   |   L (FREQUENCY)
# A   |   B    | H
# C   |   D    | L
#               (PERFORMANCE)

ROUTE_IDS = ['A', 'B', 'C', 'D']
ROUTE_PERFORMANCE_LEVELS = {'A': 'high', 'B': 'low', 'C': 'high', 'D': 'low'}
ROUTE_FREQUENCY = {'A': 'high', 'B': 'high', 'C': 'low', 'D': 'low'}

# route headway. Amything headways <10 min is "high frequency"
HEADWAYS = {'high': 10, 'low': 30}

# penalty will be used for the reward by subtracting the reward by:
# penaly * n_missing_trips
# This penalizes performance based on staffing differentially by the route frequency. This is linear but the penalty should probably
# be exponential.
# Eventually get this form an empirical distribution. If this doesnt work, try making this more aggressive, or make it time dependant
# so that the PM has a higher penalty on staffing. Can make the reward proportional to that.
PERFORMANCE_PENALTY = {
    'high': 0.1,  # high freq
    'low': 0.2  # low freq
}

# From Haris: vulnerability becomes a factor if you have different levels of it, more continuous not an attribute.
# can design a study in terms of levels and factors given the computational effort. Look this up. See if there
# is on the importance of the missed trip relative to another. Quinyi thesis looked at this problem for MBTA
# (lit review) masters thesis.
ROUTE_HEADWAYS = [
    HEADWAYS['high'], HEADWAYS['high'], HEADWAYS['low'], HEADWAYS['low']
]

# time period: 1 day (24 hours, 6 4-hour periods).
time_period_hrs = 4
time_horizon_hrs = 24

DAILY_TOTAL_EXTRABOARD = 10
MAX_MISSING_TRIPS_PCT = 0.3

DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1
LEARNING_RATE = 0.1  # how quickly you want to update the Q table. If too small, too slow. If too fast, the update function is too jerky.
LEARNING_STEPS = 30000  # 6 timesteps per episode, 30k episodes = days simulating.

NUM_STATES = 480
NUM_ACTIONS = 81  # 3*3*3*3

ASSIGNMENT_OPTIONS_PER_ROUTE = range(
    0, 3
)  # TODO: maybe alter this if I don't want to restrict actions to (0,1, 2)
TIME_PERIOD_LENGTH = 4

import numpy as np
import pandas as pd
from typing import List, Tuple

#
# H   |   L (FREQUENCY)
# A   |   B    | H
# C   |   D    | L
#               (PERFORMANCE)

ROUTE_IDS = ['A', 'B', 'C', 'D']

## we wont change it
ROUTE_PERFORMANCE_LEVELS = {'A': 'high', 'B': 'low', 'C': 'high', 'D': 'low'}

ROUTE_FREQUENCY = {'A': 'high', 'B': 'high', 'C': 'low', 'D': 'low'}

# route headway. Amything headways <10 min is "high frequency"
HEADWAYS = {'high': 10, 'low': 30}

## penalty wil be used for the reward by subtracting the reward by:
## penaly * n_missing_trips
# Eventually get this form an empirical distribution
PERFORMANCE_PENALTY = {
    'high': 0.1,  # high perf
    'low': 0.2  # low perf
}

ROUTE_HEADWAYS = [
    HEADWAYS['high'], HEADWAYS['high'], HEADWAYS['low'], HEADWAYS['low']
]

# time period: 1 day (24 hours, 6 4-hour periods). Can play with these
time_period_hrs = 4
time_horizon_hrs = 24

DAILY_TOTAL_EXTRABOARD = 30
MAX_MISSING_TRIPS_PCT = 0.3

DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1
LEARNING_RATE = 0.1
LEARNING_STEPS = 30000


NUM_STATES = 480
NUM_ACTIONS = 81
NUM_EPISODES = 30000
TIME_PERIOD_LENGTH = 4
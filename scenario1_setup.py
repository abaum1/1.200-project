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

route_headways = [
    HEADWAYS['high'], HEADWAYS['high'], HEADWAYS['low'], HEADWAYS['low']
]

# time period: 1 day (24 hours, 6 4-hour periods). Can play with these
time_period_hrs = 4
time_horizon_hrs = 24

daily_total_extraboard = 30

# scenario 1: absenteeism is uniformly distributed. Research question: How does the relationship
# between route frequency and baseline performance affect it's prioritization in extraboard allocation?
# (potentially scenario 2 is the case where absenteeism is not uniformly distributed, and we assume that performance
# is inversely correlated with absenteeism).
max_missing_trips_pct = 0.3


def get_reward(state, action):
    # we use the route type because this is not encoded in the state
    # we want the agent to "learn" which routes are more vulnerable than others
    base_reward = 1

    # for each route performance type, get the number of missing trips from the state
    # multiple each type by the respective performance penality, add together to get the total reward.

    # TODO: check that this is correct
    missing_low_perf = (state[0] + state[2]) - (
        action[0] + action[2]
    )  # the current number of missing trips on the low performance routes -
    # the trips that have been filled on those routes by the specific action
    missing_high_perf = (state[1] + state[3]) - (action[1] + action[3])
    return base_reward - (PERFORMANCE_PENALTY['high'] * missing_high_perf +
                          PERFORMANCE_PENALTY['low'] * missing_low_perf)


def pct_trips_to_missing_trips(time_period_length: int,
                               pct_missing_trips: float,
                               route_headway: int) -> int:
    # translate the pct of missing trips to number of missing trips per time period
    total_trips_per_time_period = (60 / route_headway) * time_period_length

    return int(pct_missing_trips * total_trips_per_time_period)


def generate_state(time_period_idx: int, time_period_length: int,
                   route_headways: List[int], remaining_extraboard) -> tuple:

    pct_missing_uniform = np.random.uniform(0, max_missing_trips_pct, 1)[0]
    # total_daily_extraboard = 30
    missing_trips_for_routes = [
        pct_trips_to_missing_trips(time_period_length, pct_missing_uniform,
                                   route_headway)
        for route_headway in route_headways
    ]

    return time_period_idx, *missing_trips_for_routes, remaining_extraboard


def step(prev_state, action=None):
    if action is None:
        action = list(np.random.randint(0, 2, size=4))
        print('action', action)
    used_extraboard = sum(action)
    remaining_extraboard = prev_state[-1] - used_extraboard
    time_period_idx = prev_state[0] + 1
    next_state = generate_state(time_period_idx, 4, route_headways,
                                remaining_extraboard)

    # Define the reward. Reward is the expected performance of all routes given the relationship between
    # their historical performance and level of staffing. Use synthetic data for this for now.
    return next_state


# DP will iterative over all the states and then assign a reward to every state and take the state that has
# the highest reward. We'll define the reward as well.


def naive_policy(state: Tuple) -> List:
    # returns an action
    # state: (time_step, route1, route2, route3, route4, extraboard_remaining)
    # simple policy is used to get full workflow working and also to be a benchmark to compare DP results to.
    # Naive policy is that we will assign the extraboard in order of routes with the most missing trips to the least,
    # in a greedy manner s.t. all of the missing trips of a given route are filled and we only move on to the next route
    # if there is still remaining extraboard.
    remaining_extraboard = state[-1]

    sorted_route_indices = sorted(list(enumerate(state[1:5])) , key=lambda x: x[1], reverse=True)
    indices_increasing_order = [index for index, _ in sorted_route_indices]

    action = np.zeros(4)
    for index in indices_increasing_order:
        action[index] = min(
            remaining_extraboard, state[index+1]
        )  # action is set to the number of missing trips that are assigned to that route
        remaining_extraboard = remaining_extraboard - action[index]

    return list(action)


state = generate_state(0, 4, route_headways, 30)  # generate initial state
for i in range(6):
    action = naive_policy(
        state)  # at every timestep, compute a new action based on the state
    reward = get_reward(state, action)  # get the reward of that next state
    next_state = step(
        state, action=action
    )  # get the next state as a function of the previous state and the action
    print('state', state, 'action', action, 'reward', reward, 'next_state', next_state)
    state = next_state

# scenarios
# 1. uniform % absenteeism over routes and over the course of the day
# 2. % absenteeism varies based on the route (lower performance, higher absenteeism)
# 3. % absenteeism varies based on time of day (maybe higher in the afternoon).
# True because: PM shifts more likely to be staffed by more junior operators who call out more
# and because OT is more attractive in the morning (morning missing shifts may be filled by volunteers rather
# than by extraboard).

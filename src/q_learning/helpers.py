import numpy as np
from typing import List, Dict, Tuple
from settings import PERFORMANCE_PENALTY, MAX_MISSING_TRIPS_PCT


def filter_for_valid_actions(all_actions: List[Tuple[int]],
                             state: Tuple[int]) -> List[Tuple[int]]:
    # the sum of the action cannot be greater than the remaining extraboard
    actions_less_remaining_xboard = [
        action for action in all_actions if sum(action) <= state[-1]
    ]

    # and any action index cannot be greater than the corresponding number of missing trips for the route
    filtered_actions = [
        action for action in actions_less_remaining_xboard
        if (state[1] >= action[0] and state[2] >= action[1]
            and state[3] >= action[2] and state[4] >= action[3])
    ]

    return [all_actions.index(item) for item in filtered_actions]


def generate_state(time_period_idx: int, time_period_length: int,
                   route_headways: List[int], remaining_extraboard) -> tuple:

    pct_missing_uniform = np.random.uniform(0, MAX_MISSING_TRIPS_PCT, 1)[0]
    missing_trips_for_routes = [
        pct_trips_to_missing_trips(time_period_length, pct_missing_uniform,
                                   route_headway)
        for route_headway in route_headways
    ]

    state = time_period_idx, *missing_trips_for_routes, remaining_extraboard

    return state


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

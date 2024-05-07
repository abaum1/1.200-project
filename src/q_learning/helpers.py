import numpy as np
from typing import List, Tuple
import settings
import pandas as pd
import pickle
from itertools import product


def save_q_table(q_table: pd.DataFrame, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(q_table, f)


def load_q_table(file_path: str):
    with open(file_path, 'rb') as f:
        q_table = pickle.load(f)
    return q_table


def generate_potential_actions(assignment_options_per_route: List[int]):
    return list(
        product(assignment_options_per_route, assignment_options_per_route,
                assignment_options_per_route, assignment_options_per_route))


def generate_all_states():
    time_periods = range(0, 6) 
    extraboard_possibilities = range(settings.DAILY_TOTAL_EXTRABOARD + 1)
    possible_missing_trips_route_1 = range(
        pct_trips_to_missing_trips(4, 0.3, settings.ROUTE_HEADWAYS[0]) + 1)
    possible_missing_trips_route_2 = range(
        pct_trips_to_missing_trips(4, 0.3, settings.ROUTE_HEADWAYS[1]) + 1)
    possible_missing_trips_route_3 = range(
        pct_trips_to_missing_trips(4, 0.3, settings.ROUTE_HEADWAYS[2]) + 1)
    possible_missing_trips_route_4 = range(
        pct_trips_to_missing_trips(4, 0.3, settings.ROUTE_HEADWAYS[3]) + 1)

    return list(
        product(time_periods, possible_missing_trips_route_1,
                possible_missing_trips_route_2, possible_missing_trips_route_3,
                possible_missing_trips_route_4, extraboard_possibilities))


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

    pct_missing_uniform = np.random.uniform(0, settings.MAX_MISSING_TRIPS_PCT,
                                            1)[0]
    missing_trips_for_routes = [
        pct_trips_to_missing_trips(time_period_length, pct_missing_uniform,
                                   route_headway)
        for route_headway in route_headways
    ]

    state = time_period_idx, *missing_trips_for_routes, remaining_extraboard

    return state


def pct_trips_to_missing_trips(time_period_length: int,
                               pct_missing_trips: float,
                               route_headway: int) -> int:
    # translate the pct of missing trips to number of missing trips per time period
    total_trips_per_time_period = (60 / route_headway) * time_period_length

    return int(pct_missing_trips * total_trips_per_time_period)

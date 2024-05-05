import q_learning.helpers as helpers
import settings
from collections import Counter
from itertools import product
import math

# random_states = [generate_state(0, 4, route_headways, 30)[1:5]
#  for _ in range(200)]  # idx, #time periodlength, remaining extraboard

# counts = Counter(random_states)
# print(counts)
# total_counts = sum(counts.values())

# probabilities = {key: count / total_counts for key, count in counts.items()}
# print(probabilities.values())

#generate all the potential states:

# 6 * missing options 1 * missing options 2 * missing options 3 * missing options 4 * num extraboard

# round up max for each of the routes as a fn of their freq and MAX_MISSING_TRIPS_PCT = 0.3
# then for each route get the max potential missing trips, add 4 more for loops one for erach route iterating
# through 0, max number.
# look at itertools.product


def generate_all_states():
    time_periods = range(0, 6) # decided to make states 0 indexed
    extraboard_possibilities = range(settings.DAILY_TOTAL_EXTRABOARD + 1)
    possible_missing_trips_route_1 = range(
        helpers.pct_trips_to_missing_trips(4, 0.3, settings.ROUTE_HEADWAYS[0])
        + 1)
    possible_missing_trips_route_2 = range(
        helpers.pct_trips_to_missing_trips(4, 0.3, settings.ROUTE_HEADWAYS[1])
        + 1)
    possible_missing_trips_route_3 = range(
        helpers.pct_trips_to_missing_trips(4, 0.3, settings.ROUTE_HEADWAYS[2])
        + 1)
    possible_missing_trips_route_4 = range(
        helpers.pct_trips_to_missing_trips(4, 0.3, settings.ROUTE_HEADWAYS[3])
        + 1)

    return list(
        product(time_periods, possible_missing_trips_route_1,
                possible_missing_trips_route_2, possible_missing_trips_route_3,
                possible_missing_trips_route_4, extraboard_possibilities))



# all_states = generate_all_states()
# print('num states', len(list(all_states)))


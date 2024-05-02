import q_learning.helpers as helpers
import settings
from collections import Counter

# random_states = [generate_state(0, 4, route_headways, 30)[1:5]
#  for _ in range(200)]  # idx, #time periodlength, remaining extraboard

# counts = Counter(random_states)
# print(counts)
# total_counts = sum(counts.values())

# probabilities = {key: count / total_counts for key, count in counts.items()}
# print(probabilities.values())

#generate all the potential states:

# 6 * missing options 1 * missing options 2 * missing options 3 * missing options 4 * num extraboard


def generate_all_states():
    states = []
    time_period_idxs = range(1, 7)
    remaining_extraboards = range(
        settings.DAILY_TOTAL_EXTRABOARD
    )  # Assuming you want to include 30 as an option

    # Generate all possible combinations of time_period_idx and remaining_extraboard
    for time_period_idx in time_period_idxs:
        for remaining_extraboard in remaining_extraboards:
            # Generate all possible combinations of route_headways
            # For this example, let's assume we have 3 routes with headways [10, 20, 30]
            # route_headways = [10, 20, 30]  # Example route headways
            # missing_trips_for_routes = [
            #     helpers.pct_trips_to_missing_trips(
            #         time_period_idx, settings.MAX_MISSING_TRIPS_PCT,
            #         route_headway) for route_headway in route_headways
            # ]
            # Generate state for each combination
            state = helpers.generate_state(time_period_idx,
                                           len(settings.ROUTE_HEADWAYS),
                                           settings.ROUTE_HEADWAYS,
                                           remaining_extraboard)
            states.append(state)
    return states



all_states = generate_all_states()
print('num states', len(list(all_states)), list(all_states))

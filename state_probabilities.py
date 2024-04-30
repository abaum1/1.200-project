from scenario1_setup import generate_state, route_headways

from collections import Counter

random_states = [generate_state(0, 4, route_headways, 30)[1:5]
 for _ in range(200)]  # idx, #time periodlength, remaining extraboard

counts = Counter(random_states)
print(counts)
total_counts = sum(counts.values())

probabilities = {key: count / total_counts for key, count in counts.items()}
print(probabilities.values())

# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

# load q-table from file
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
except Exception as e:
    print("Error loading Q-table:", e)
    q_table = {}

# Global flag for whether the passenger has been picked up.
passenger_in_taxi = [False]
# Global history: a list that will store tuples of (action, reward)
action_reward_history = []


def get_state_key(state, passenger_in_taxi, last_action=None, last_reward=None, pickup_action_code=4):
    taxi_row, taxi_col = state[0], state[1]
    
    # Get the fixed station positions.
    stations = [
        (state[2], state[3]),
        (state[4], state[5]),
        (state[6], state[7]),
        (state[8], state[9])
    ]
    
    # Obstacles: north, south, east, west.
    obstacles = state[10:14]
    
    # The raw flags from the state.
    passenger_look = state[14]
    destination_look = state[15]
    
    # Update the passenger flag based on the last action if it was a pickup.
    # (Remember: a valid pickup yields a reward greater than -10.)
    if passenger_in_taxi[0] == False:
        if last_action is not None and last_action == pickup_action_code:
            if last_reward is not None and last_reward > -10:
                passenger_in_taxi[0] = True
    
    # Compute Manhattan distances from the taxi to each station.
    station_distances = tuple(abs(taxi_row - s[0]) + abs(taxi_col - s[1]) for s in stations)
    
    # Identify candidate destination station(s):
    # When destination_look is true, the destination must be one of the stations adjacent to or at the taxi.
    candidate_destinations = []
    if destination_look:
        for idx, s in enumerate(stations):
            if abs(taxi_row - s[0]) + abs(taxi_col - s[1]) <= 1:
                candidate_destinations.append(idx)
    else:
        # Otherwise, consider all stations as potential (this part is less informative, but necessary).
        candidate_destinations = list(range(4))
    
    # Build the key:
    # We flatten all the important information into one tuple.
    key = (taxi_row, taxi_col, int(passenger_in_taxi[0]), destination_look) \
          + obstacles \
          + station_distances \
          + (tuple(candidate_destinations),)
    
    return key

def get_action(obs):
    global q_table, passenger_in_taxi, action_reward_history

    if action_reward_history:
        last_action, last_reward = action_reward_history[-1]
    else:
        last_action, last_reward = None, None

    state_key = get_state_key(obs, passenger_in_taxi, last_action, last_reward)
    
    if state_key in q_table:
        if np.random.random() < 0.1: # give flexibility in test time
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = int(np.argmax(q_table[state_key]))
    else:
        action = random.choice([0, 1, 2, 3, 4, 5])

    return action
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
passenger_in_taxi = False
phase = 1                     # 1: pickup phase; 2: dropoff phase
target_station_index = 0  


def get_state_key(state, target_pos, passenger_in_taxi):
    taxi_row, taxi_col = state[0], state[1]

    # Extract obstacle information from state
    obstacle_north = state[10]
    obstacle_south = state[11]
    obstacle_east = state[12]
    obstacle_west = state[13]

    # Compute dx, dy from taxi to the target position
    dx_target = target_pos[0] - taxi_row
    dy_target = target_pos[1] - taxi_col

    passenger_look = state[-2]
    destination_look = state[-1]

    key = (dx_target, dy_target, passenger_in_taxi, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
    return key


def get_action(obs):
    global q_table, passenger_in_taxi, phase, target_station_index

    # Extract station positions from obs (indices 2-9)
    stations = [(obs[2], obs[3]),
                (obs[4], obs[5]),
                (obs[6], obs[7]),
                (obs[8], obs[9])]
    
    state_key = get_state_key(obs, stations[target_station_index], passenger_in_taxi)
    if state_key in q_table:
        if np.random.random() < 0.3: # give flexibility in test time
            # print(f"have state_key, but choose random for flexibility")
            action = random.choice([0, 1, 2, 3])
        else:
            action = int(np.argmax(q_table[state_key]))
            # print(f"have state key, greedy action")
        # action = int(np.argmax(q_table[state_key]))
        # print(f"have state key, greedy action")
    else:
        action = random.choice([0, 1, 2, 3])
        # print(f"no state key, random action")
    # print(f"state_key: {state_key}, action: {action}")

    
    # ----- Phase Transition -----
    distance_to_target = abs(state_key[0]) + abs(state_key[1])
    destination_look = state_key[-1]
    passenger_look = state_key[-2]

    if phase == 1: # pick up phase
        if distance_to_target == 0:
            if passenger_look == False:
                target_station_index = (target_station_index + 1) % 4 # try other passenger station
            else:
                if action == 4: # pick up
                    passenger_in_taxi = True
                    phase = 2 # switch to phase 2
                    target_station_index = (target_station_index + 1) % 4 # move to destination station
        elif distance_to_target == 1:
            if passenger_look == False: # not correct station
                target_station_index = (target_station_index + 1) % 4 # try other passenger station

    elif phase == 2: # drop off phase
        if distance_to_target == 0:
            if destination_look == False:
                if action == 5: # accidental drop off at wrong station => move back to phase 1
                    phase = 1
                    passenger_in_taxi = False
                    # remain at same target_station_index
                else:
                    target_station_index = (target_station_index + 1) % 4 # try other destination station
        elif distance_to_target == 1:
            if destination_look == False:
                target_station_index = (target_station_index + 1) % 4 # try other destination station

    # print(f"phase: {phase}, passenger_in_taxi: {passenger_in_taxi}, target_station_index: {target_station_index}")

    return action
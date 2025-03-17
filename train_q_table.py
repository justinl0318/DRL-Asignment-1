import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from simple_custom_taxi_env import SimpleTaxiEnv

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


def train_q_table(num_episodes=1000, max_steps=5000, alpha=0.2, gamma=0.95,
                  epsilon_start=0.95, epsilon_end=0.10, epsilon_decay=0.9999):


    q_table = {}  # dictionary mapping custom state keys to Q-values for each action
    num_actions = 6  # Actions: 0:South, 1:North, 2:East, 3:West, 4:Pickup, 5:Dropoff
    epsilon = epsilon_start
    rewards_per_episode = []
    done_per_episode = []
    phase2_per_episode = []

    for episode in range(num_episodes):
        env = SimpleTaxiEnv(grid_size=random.randint(5, 10), fuel_limit=5000)
        passenger_in_taxi = False # to know whether the taxi has picked up the agent, reset at start of each episode
        phase = 1  # 1: pickup passenger phase, 2: dropoff passenger at destination phase
        target_station_index = 0
        passenger_position = None # unknown at first, known when the taxi pick it up at a station

        # IDEA: at phase 1, we try each station sequentially. If we arrive at a station and passenger_look is true,
        # that means we should pick up and move to phase 2. Otherwise, we try to go to the next station. 
        # at phase 2, we try the following stations sequentially too. If we arrivate at a station and destination_look is true,
        # we should drop the passenger. Otherwise, we try to go to the next station.

        # Reset environment and get initial state.
        state, _ = env.reset()
        # Extract station positions from state (indices 2-9)
        stations = [(state[2], state[3]),
                    (state[4], state[5]),
                    (state[6], state[7]),
                    (state[8], state[9])]
        
        # phase 1
        state_key = get_state_key(state, stations[target_station_index], passenger_in_taxi)

        total_reward = 0

        for step in range(max_steps):
            # Initialize Q-values for unseen state keys
            if state_key not in q_table:
                q_table[state_key] = np.zeros(num_actions)
                
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randrange(num_actions)
            else:
                action = np.argmax(q_table[state_key])
    
            next_state, reward, done, _ = env.step(action)

            switched_target_station_index = False

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
                            passenger_position = (state[0], state[1])
                            phase = 2 # switch to phase 2
                            phase2_per_episode.append(step)
                            switched_target_station_index = True
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
                        
                # if distance_to_target <= 1 and destination_look == False:
                #     target_station_index = (target_station_index + 1) % 4 # try other destination station

            next_state_key = get_state_key(next_state, stations[target_station_index], passenger_in_taxi)
            # ----- End of Phase Transition -----

            
            # ----- Reward Shaping -----
            shaped_reward = 0
            if action == 4: # pick up
                if passenger_in_taxi == False: 
                    if distance_to_target == 0 and passenger_look: # pick up at correct station
                        shaped_reward += 10
                    else: # pick up but not at station or not at correct station => penalty
                        shaped_reward -= 10
                else: # passenger in taxi but pick up again => penalty
                    shaped_reward -= 10
            if action == 5: # drop off
                if passenger_in_taxi:
                    if distance_to_target == 0 and destination_look: # drop off at correct station, which is the destination
                        shaped_reward += 100
                    else: # drop off at the wrong destination
                        shaped_reward -= 200
                else: # passenger not in taxi
                    shaped_reward -= 10

            current_distance = abs(state_key[0]) + abs(state_key[1])
            new_distance = abs(next_state_key[0]) + abs(next_state_key[1])
            if switched_target_station_index == False:
                shaped_reward += (current_distance - new_distance) * 1

            # obstacle_north = state_key[-4]
            # obstacle_south = state_key[-3]
            # obstacle_east = state_key[-2]
            # obstacle_west = state_key[-1]
            # if (obstacle_north and action == 1) or (obstacle_south and action == 0) or (obstacle_east and action == 2) or (obstacle_west and action == 3):
            #     shaped_reward -= 10
            
            # ----- End of reward shaping -----

            reward += shaped_reward
            total_reward += reward

            if next_state_key not in q_table:
                q_table[next_state_key] = np.zeros(num_actions)

            # Q-learning update rule:
            best_next = np.max(q_table[next_state_key])
            q_table[state_key][action] = q_table[state_key][action] + alpha * (reward + gamma * best_next - q_table[state_key][action])


            state_key = next_state_key
            state = next_state

            if done:
                done_per_episode.append(step)
                break
        
        # Decay epsilon after each episode
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            avg_phase2_step = np.mean(phase2_per_episode[-100:])
            avg_done_step = np.mean(done_per_episode[-100:])
            print(f"🚀 Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
            print(f"🚀 Episode {episode + 1}/{num_episodes}, Average phase2 turn step: {avg_phase2_step:.2f}")
            print(f"🚀 Episode {episode + 1}/{num_episodes}, Average done step: {avg_done_step:.2f}")


    # Save the trained Q-table to a file
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Training complete. Q-table saved as q_table.pkl")

    return rewards_per_episode, phase2_per_episode, done_per_episode


if __name__ == "__main__":
    rewards, phase2, done = train_q_table(num_episodes=150000, max_steps=3000)
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Reward Training Progress")
    plt.savefig("reward_plot.png")  # Save as PNG
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(phase2, label="Phase2 Turn Steps per Episode")
    plt.plot(done, label="Done Steps per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.title("Phase2 Turn Steps & Done Steps per Episode")
    plt.legend()
    plt.savefig("phase2_done_plot.png")
    plt.close()

    print("Plot saved as reward_plot.png")

import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from simple_custom_taxi_env import SimpleTaxiEnv

def get_state_key(state, passenger_in_taxi, last_action=None, last_reward=None, pickup_action_code=4):
    """
    Converts the raw state tuple from get_state() into a state key for the Q-table.
    
    Parameters:
      state: A tuple with the following format:
             (taxi_row, taxi_col,
              station0_row, station0_col,
              station1_row, station1_col,
              station2_row, station2_col,
              station3_row, station3_col,
              obstacle_north, obstacle_south, obstacle_east, obstacle_west,
              passenger_look, destination_look)
      passenger_in_taxi: A boolean flag wrapped in list that is False until a 
      successful PICKUP occurs. Once set to True, it remains True 
      for the rest of the episode.              
      last_action: The action taken on the previous step (if any).
      last_reward: The reward obtained on the previous action (if any).

      pickup_action_code: The integer code for the PICKUP action (default: 4).
      
    Returns:
      key: A tuple that encodes the state for the Q-table.
           The key includes:
             - Taxi position (row, col)
             - Passenger flag (0: waiting, 1: onboard)
             - The raw destination look flag
             - Obstacle flags (north, south, east, west)
             - Manhattan distances from taxi to each of the 4 station positions
             - Candidate destination indexes, if the taxi is near a station (when destination_look is True)
    """
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



def train_q_table(num_episodes=1000, max_steps=5000, alpha=0.2, gamma=0.99,
                  epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999):

    env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)
    q_table = {}  # dictionary mapping custom state keys to Q-values for each action
    num_actions = 6  # Actions: 0:South, 1:North, 2:East, 3:West, 4:Pickup, 5:Dropoff

    epsilon = epsilon_start
    rewards_per_episode = []
    for episode in range(num_episodes):
        passenger_in_taxi = [False] # to know whether the taxi has picked up the agent, reset at start of each episode

        state, _ = env.reset()
        state_key = get_state_key(state, passenger_in_taxi)
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
            next_state_key = get_state_key(next_state, passenger_in_taxi, action, reward)
            
            if next_state_key not in q_table:
                q_table[next_state_key] = np.zeros(num_actions)
            
            # Q-learning update rule:
            best_next = np.max(q_table[next_state_key])
            q_table[state_key][action] = q_table[state_key][action] + alpha * (reward + gamma * best_next - q_table[state_key][action])
            
            state_key = next_state_key
            total_reward += reward
            
            if done:
                break
        
        # Decay epsilon after each episode
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"🚀 Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")


    # Save the trained Q-table to a file
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Training complete. Q-table saved as q_table.pkl")

    return rewards_per_episode


if __name__ == "__main__":
    rewards = train_q_table(num_episodes=5000)
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Reward Training Progress")
    plt.savefig("reward_plot.png")  # Save as PNG
    #plt.show() #comment out the show function.
    print("Plot saved as reward_plot.png")

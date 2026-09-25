# src/main.py

import yaml
from environment import WindFarmEnv
from agent import create_actor_network, create_critic_network

TERRAIN_FILE = 'data/terrain.tif'
WIND_DATA_FILE = 'data/wind_data.lib'

if __name__ == '__main__':
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 1. Initialize the environment
    env = WindFarmEnv(terrain_path=TERRAIN_FILE, wind_data_path=WIND_DATA_FILE, config=config)

    # 2. Create the Actor and Critic networks
    actor_model = create_actor_network(env.observation_space, env.action_space)
    critic_model = create_critic_network(env.observation_space, env.action_space)

    # 3. Print the summaries to verify the architecture
    print("\n--- Actor Network Summary ---")
    actor_model.summary()

    print("\n--- Critic Network Summary ---")
    critic_model.summary()
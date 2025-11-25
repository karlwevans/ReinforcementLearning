# Re-run this cell to install and import the necessary libraries and load the required variables
import numpy as np
import gymnasium as gym
import imageio
from IPython.display import Image
from gymnasium.utils import seeding

# Initialize the Taxi-v3 environment
env = gym.make("Taxi-v3", render_mode='rgb_array')

# Seed the environment for reproducibility
env.np_random, _ = seeding.np_random(42)
env.action_space.seed(42)
np.random.seed(42)

# Maximum number of actions per training episode
max_actions = 100 

#1 Train Agent
num_episodes = 2000
alpha = 0.1
gamma = 1
epsilon=0.2
num_states, num_actions = env.observation_space.n, env.action_space.n
q_table = np.zeros((num_states, num_actions))

def epsilon_greedy(state):
    # Implement the condition to explore
    if np.random.rand() < epsilon:
      	# Choose a random action
        action = env.action_space.sample()
    else:
      	# Choose the best action according to q_table
        action = np.argmax(q_table[state, :]) 
    return action

def update_q_table(state, action, reward, new_state):  
    old_value = Q[state, action]  
    next_max = max(Q[new_state])  
    q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

episode_returns = []
for episode in range(num_episodes):    
    state, info = env.reset()    
    # terminated = False    
    episode_reward = 0
    # while not terminated:
    for i in range(max_actions):
        # Random action selection        
        action =  epsilon_greedy(state)
        # Take action and observe new state and reward        
        new_state, reward, terminated, truncated, info = env.step(action)
        # Update Q-table        
        episode_reward += reward    
        update_q_table(state, action, reward,new_state)          
        state = new_state 
    episode_returns.append(episode_reward)



#2 Analyze
policy = {state: np.argmax(Q[state]) for state in range(num_states)}


#3 Test Policy
frames=[]
state, info = env.reset(seed=42)    
episode_total_reward = 0
    #while not terminated:
i=1
while i <=16:
    # Select the best action based on learned Q-table        
    action = policy[state]
    # Take action and observe new state        
    new_state, reward, terminated, truncated, info = env.step(action)        
    state = new_state        
    episode_total_reward += reward
    frames.append(env.render())
    i+=1

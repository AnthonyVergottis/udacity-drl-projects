from unityagents import UnityEnvironment
import numpy as np

import random
import time
import numpy as np
import torch

from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import DDPG_AGENT

env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]


agent = DDPG_AGENT(state_size, action_size, 0)

def train_agent(n_episodes=2000, max_t=2000, print_every=10):
    scores_deque = deque(maxlen=100)
    scores = []
    
    for i_episode in range(1, n_episodes+1):
        
        # Reset Env and Agent
        env_info = env.reset(train_mode=True)[brain_name]  
        states = env_info.vector_observations                   
        episode_scores = np.zeros(num_agents)                           
        agent.reset()
                
        for t in range(max_t):
            actions = agent.act(states)
            
            env_info = env.step(actions)[brain_name]           
            next_states = env_info.vector_observations         
            rewards = env_info.rewards                         
            
            dones = env_info.local_done                        


            # Add experiences to replay buffer
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done) 
            
            episode_scores += env_info.rewards                          
            states = next_states                                 
           
            
            if np.any(dones):                                  
                break
     
        # Find Maximum score of both agents and add to array
        scores_deque.append(np.max(episode_scores))
        scores.append(np.max(episode_scores))
        
        if i_episode % print_every == 0:
           print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        
        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    
    return scores

scores = train_agent()


# Plot scores over episodes
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('images/ddpg_scores.png', bbox_inches='tight')
plt.show()

# Load the saved weights into Pytorch model
# agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth', map_location='cpu'))
# agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth', map_location='cpu'))

# for i in range(100):                                         # play game for 5 episodes
#     env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
#     states = env_info.vector_observations                  # get the current state (for each agent)
#     scores = np.zeros(num_agents)                          # initialize the score (for each agent)
#     while True:
#         actions = agent.act(states)                        # select actions from loaded model agent
#         actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#         env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#         next_states = env_info.vector_observations         # get next state (for each agent)
#         rewards = env_info.rewards                         # get reward (for each agent)
#         dones = env_info.local_done                        # see if episode finished
#         scores += env_info.rewards                         # update the score (for each agent)
#         states = next_states                               # roll over states to next time step
#         if np.any(dones):                                  # exit loop if episode finished
#             break
#     print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

env.close()
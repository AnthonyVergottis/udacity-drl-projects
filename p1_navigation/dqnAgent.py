import numpy as np
import random
from collections import namedtuple, deque

from neuralNetwork import neuralNet

import torch
import torch.nn.functional as F
import torch.optim as optim

# Some Model Hyperparameter
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

# Command to check if GPU is present on system
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    #
    # Initialise AI agent
    #
    def __init__(self,state_size,action_size,seed):
        #
        # Params:
        # state_size (int) = dimensions of state vector
        # action_size (int) = dimensions of action vector
        # seed (int) = seed
        #
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        #
        # Cretea Target and Local Neural Networks
        # networkTarget = Target Network
        # networkLocal = Local Network
        # optimizer = Algorithm used to perform gradient descent
        #
        self.networkTarget = neuralNet(state_size,action_size,seed).to(device)
        self.networkLocal  = neuralNet(state_size,action_size,seed).to(device)
        self.optimizer = optim.Adam(self.networkLocal.parameters(), lr=LR)
        #
        # Create a Replay memory buffer
        #
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initial time step set to zero, used to update network at every
        # UPDATE_EVERY time step
        self.t_step = 0

    def getAction(self,state,eps=0.0):
        #
        # Select action to take given state
        # state (array) = Current states
        # eps (float) = epsilon, for epsilon greedy action selection
        #
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.networkLocal.eval()
        with torch.no_grad():
            action_values = self.networkLocal(state)
            # print(action_values.shape)
        self.networkLocal.train()

        # Select action using epsilon-greedy
        if random.random() > eps:
            # print(np.argmax(action_values.cpu().data.numpy()))
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # print(random.choice(np.arange(self.action_size)))
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        # print("Actions Size: ", actions.shape)
        # print("Actions : ", actions)
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.networkTarget(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))


        # Get expected Q values from local model
        Q_expected = self.networkLocal(states).gather(1, actions)
        # print("Q_Expected Size: ", Q_expected)
        # print(Q_expected.shape)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.networkLocal, self.networkTarget, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def step(self,state,action,reward,next_state,done):
        # Add step to replay memory
        self.memory.add(state,action,reward,next_state,done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # print(states)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

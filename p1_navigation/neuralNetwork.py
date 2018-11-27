import torch
import torch.nn as nn
import torch.nn.functional as F

class neuralNet(nn.Module):
    # Initialise Neural Network architecture
    #
    # Params:
    # state_size (int) = dimensions of state vector
    # action_size (int) = dimensions of action vector
    # seed (int) = seed
    # fc1_units (int) = number of neurons in first hidden layer
    # fc2_units (int) = number of neurons in second hidden layer
    #
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64):
        super(neuralNet,self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

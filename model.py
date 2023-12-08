import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_size:int, action_size:int):
        '''
        Initialize the actor network.
        It receive the observation for a single actor in the shape (..., state_size), 
        and use the Tanh activation function to limit the output in the range (-1, 1).

        Params:
            state_size: the last dim size for the observation.
            action_size: the last dim size for the action.
        '''
        super().__init__()
        # initiate network architecture
        self.input_projector = nn.Linear(state_size, 128)
        self.fc_mu = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Tanh()
        )
        self.action_size = action_size


    def forward(self, state:torch.Tensor, std:float):
        '''
        transform the local observation into a multivariate normal distribution, 
        which will be used to sample the action and generate log probability for a certain 
        actor.

        Params:
            state: the local observation for a certain actor.
            std: the manual-setted standard deviation for multivariate normal distribution.

        Return:
            A multivariate normal distribution which will be used to sample the action and 
            generate log probability.
        '''
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(DEVICE)
        x = self.input_projector(state)
        # calculate mu value for normal distribution
        mu = self.fc_mu(x)
        # transform sigma value for normal distribution
        sigma = torch.tensor(std).to(DEVICE)
        # generate distribution
        return MultivariateNormal(mu, torch.diag_embed(sigma.expand_as(mu)))
    

class Critic(nn.Module):
    def __init__(self, state_size:int) -> None:
        '''
        Initialize the critic network.
        It receive the global state in the shape (..., state_size * num_agent), 
        return the value for the state in the shape (..., 1).

        Params:
            state_size: the concatenated state (or global state) size for critic.
        '''
        super().__init__()
        self.state_input_projector = nn.Linear(state_size, 128)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state:torch.Tensor):
        '''
        Generate the centralized value for the global state.

        Params:
            state: flattened global state in the shape (..., state_size * num_agent).

        Return:
            the centralized value for the global state.
        '''
        # transform state input
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(DEVICE)
        state = self.state_input_projector(state)
        # transform action input
        v = self.fc(state)
        return v

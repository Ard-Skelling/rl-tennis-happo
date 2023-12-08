import sys
from pathlib import Path

ABS_PATH = Path(__file__).parent
sys.path.append(str(ABS_PATH))

import random
import numpy as np
import torch
from collections import deque
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

# import the local actor and critic
from model import Actor, Critic


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ACTOR_LR = 5e-6
CRITIC_LR = 5e-5
BATCH_SIZE = 128
BUFFER_SIZE = 2048
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_FACTOR = 0.18
NUM_EPOCH = 8
UPDATE_PER_STEP = 128

TAU_ACTOR = 1e-3
TAU_CRITIC = 1e-3

STD = 0.01
    

class RolloutBuffer:
    '''
    The rollout maintain a rollout buffer, which collects (state, action, reward, done) trajectories.
    '''
    def __init__(self, buffer_size:int, num_agent:int, state_size:int, action_size:int) -> None:
        '''
        Initialize buffer instance attributes and reset the buffer.

        Params:
            buffer_size: the max time step volumn for the buffer.
            num_agent: the amount of agents.
            state_size: the last dim size for state.
            action_size: the last dim size for action.
        '''
        self.buffer_size = buffer_size
        self.num_agent = num_agent
        self.state_size = state_size
        self.action_size = action_size
        self.reset()

    def reset(self):
        '''
        Reset the buffer attributes, empty all experiences.
        '''
        self.state_mem = np.zeros((self.buffer_size, self.num_agent, self.state_size), dtype=np.float32)
        self.action_mem = np.zeros((self.buffer_size, self.num_agent, self.action_size), dtype=np.float32)
        self.value_mem = np.zeros((self.buffer_size), dtype=np.float32)
        self.log_prob_mem = np.zeros((self.buffer_size, self.num_agent), dtype=np.float32)
        self.reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
        self.done_mem = np.zeros((self.buffer_size), dtype=np.int8)
        self.advantage_mem = np.zeros((self.buffer_size), dtype=np.float32)
        self.current_next_state = None
        self.is_full = False
        self.pos = 0

    def calculate_gae_advantage(self, values:np.ndarray, critic_model:nn.Module, \
            gamma:float=GAMMA, gae_lambda:float=GAE_LAMBDA):
        '''
        calculate GAE advantage. https://arxiv.org/abs/1506.02438
        Params:
            values: the values for self.states_mem series.
            critic_model: the critic network instance used to predict value for the current next state.
            gamma: the discount rate for value, reward or advantage calculation.
            gae_lambda: the weight for GAE advantage calculation to balance the bias and variance trade-off.
        Returns:
            the GAE advantages sequence.
        '''
        rewards = self.reward_mem
        dones = self.done_mem
        # initiate advantages, in shape (time step, num_agent)
        advantages = np.zeros(dones.shape)
        # evaluate the current next value with critic for advantage calculation, the details are in ./Report.md
        with torch.no_grad():
            last_value = critic_model(self.current_next_state.flatten()).squeeze().detach().cpu().numpy()
        full_values = np.append(values, np.expand_dims(last_value, axis=0), axis=0)
        # initiate advantage for one time step
        adv_t = 0.
        for t in reversed(range(len(dones))):
            delta = rewards[t] + gamma * full_values[t + 1] * (1 - dones[t]) - full_values[t]
            # apply GAE decay
            adv_t = delta + gamma * gae_lambda * adv_t * (1 - dones[t])
            advantages[t] = adv_t
        return advantages

    def add(self, state:np.ndarray, action:np.ndarray, log_prob:np.ndarray, reward:np.ndarray, \
            done:np.ndarray, next_state:np.ndarray):
        '''
        add a batch of time step experience into buffer.
        '''
        self.pos = min(self.pos + 1, BUFFER_SIZE)
        if self.pos == self.buffer_size:
            self.is_full = True
        for element in ['state_mem', 'action_mem', 'log_prob_mem', 'reward_mem', 'done_mem', 'advantage_mem']:
            self.__dict__[element][:-1] = self.__dict__[element][1:]
        self.state_mem[-1] = state
        # print(self.action_mem.shape)
        self.action_mem[-1] = action
        self.log_prob_mem[-1] = log_prob
        self.reward_mem[-1] = reward
        self.done_mem[-1] = done
        self.current_next_state = torch.from_numpy(next_state).float().to(DEVICE)
        return self.is_full

    def prepare_data(self, actor_0:nn.Module, actor_1:nn.Module, critic_model:nn.Module):
        '''
        Calculate the values, log probabilities and advantages for all experiences in buffer, with the 
        current actor and critic model parameters.

        Params:
            actor_0: the first actor
            actor_1: the second actor
            critic_model: the global critic model

        Return:
            All states, actions, values, log_probs, advantages in buffer and the length of bufer.
        '''
        states = torch.from_numpy(self.state_mem).float().to(DEVICE)
        actions = torch.from_numpy(self.action_mem).float().to(DEVICE)
        # calculate the current values for states and log probabilities for old actions, as the model parameters has been updated,
        # so as the distributions.
        with torch.no_grad():
            values = critic_model(states.flatten(-2, -1)).detach().squeeze()
            dist_0 = actor_0(states[:, 0, :].squeeze(), STD)
            dist_1 = actor_1(states[:, 1, :].squeeze(), STD)
            log_probs_0 = dist_0.log_prob(actions[:, 0, :].squeeze())
            log_probs_1 = dist_1.log_prob(actions[:, 1, :].squeeze())
        log_probs = torch.stack([log_probs_0, log_probs_1], dim=1)
        # calculate advantages
        advantages = self.calculate_gae_advantage(values.cpu().numpy(), critic_model, gamma=GAMMA, gae_lambda=GAE_LAMBDA)
        advantages = torch.from_numpy(advantages).float().to(DEVICE)
        memory_amout = advantages.shape[0]
        assert memory_amout >= BATCH_SIZE, \
            f'Not enough experiences. Experiences num: {memory_amout}'
        return states, actions, values, log_probs, advantages, memory_amout
        
    def sample(self, actor_0, actor_1, critic_model, batch_size=None):
        '''
        Sample a mini batch in batch_size for training.

        Params:
            actor_0: the first actor.
            actor_1: the second actor.
            critic_model: the global critic model.
            batch_size: the size of mini batch.

        Return:
            the shuffled experiences for PPO algorithm training.
        '''
        states, actions, values, log_probs, advantages, memory_amout = self.prepare_data(actor_0, actor_1, critic_model)
        batch_size = batch_size or BATCH_SIZE
        indices = np.random.choice(np.arange(memory_amout), size=batch_size)
        return states[indices], actions[indices], values[indices], log_probs[indices], advantages[indices]

class Agent:
    def __init__(self, state_size, action_size, seed=996):

        # initalize actor
        self.actor_0 = Actor(state_size, action_size).to(DEVICE)
        self.actor_1 = Actor(state_size, action_size).to(DEVICE)
        self.actor_0_opt = Adam(self.actor_0.parameters(), lr=ACTOR_LR)
        self.actor_1_opt = Adam(self.actor_1.parameters(), lr=ACTOR_LR)

        # initalize critic
        self.critic_local = Critic(state_size * 2).to(DEVICE)
        self.critic_opt = Adam(self.critic_local.parameters(), lr=CRITIC_LR)

        # initalize scheduler for learning rate decay in the training
        self.actor_0_scheduler = LinearLR(self.actor_0_opt, start_factor=1., end_factor=0.5, total_iters=40000)
        self.actor_1_scheduler = LinearLR(self.actor_1_opt, start_factor=1., end_factor=0.5, total_iters=40000)
        self.critic_scheduler = LinearLR(self.critic_opt, start_factor=1., end_factor=0.5, total_iters=40000)

        # intialize replay buffer
        self.memory = RolloutBuffer(buffer_size=BUFFER_SIZE, num_agent=2, state_size=state_size, action_size=action_size)

        # initialize noise for exploration
        self.noise = OUNoise((action_size), seed=seed)

        # the flag to triger update by every UPDATE_PER_STEP time steps.
        self.update_flag = 0

    def act(self, state:np.ndarray):
        '''
        Generate the deterministic action for evaluation/perdiction.

        Params:
            state: the whole observation for all actors, in shape (num_agent, state_size).

        Return:
            actions used to interact with environment.
        '''
        x = torch.from_numpy(state).float().to(DEVICE)
        with torch.no_grad():
            dist_0: MultivariateNormal = self.actor_0(x[0], 1e-8)
            dist_1: MultivariateNormal = self.actor_1(x[1], 1e-8)
        a0 = dist_0.mode()
        a1 = dist_1.mode()
        action = torch.clamp(torch.stack([a0, a1]), -1., 1.)
        return action.cpu().numpy()

    def generate_action(self, state:np.ndarray):
        '''
        Generate the sampled action and log probability for training.

        Params:
            state: the whole observation for all actors, in shape (num_agent, state_size).

        Return:
            actions used to interact with environment and the log_probabilities in shape (num_agent,) for PPO training.
        '''
        x = torch.from_numpy(state).float().to(DEVICE)
        # get the Normal distributions to sample the actions
        with torch.no_grad():
            dist_0: MultivariateNormal = self.actor_0(x[0], STD)
            dist_1: MultivariateNormal = self.actor_1(x[1], STD)
        a0 = dist_0.sample()
        a0 += self.noise.sample()
        a1 = dist_1.sample()
        a1 += self.noise.sample()
        action = torch.clamp(torch.stack([a0, a1]), -1., 1.)
        # print(action.shape)
        # calculate the log probability
        log_prob_0 = dist_0.log_prob(action[0])
        log_prob_1 = dist_1.log_prob(action[1])
        log_prob = torch.stack([log_prob_0, log_prob_1])
        # print(action)
        # time.sleep(1)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def step(self, state:np.ndarray, action:np.ndarray, log_prob:np.ndarray, reward:np.ndarray, \
            done:np.ndarray, next_state:np.ndarray):
        '''
        Add the experience of one time step into memory(buffer), if the traing condition is triggered,
        update the model parameters with PPO algorithm.
        The update will be triggered every UPDATE_PER_STEP time steps if the memory(buffer) is full.
        The model params will be updated NUM_EPOCH times for each updating operation.

        Params:
            state: full state in shape (num_agent, state_size)
            action: stacked action in shape (num_agent, action_size)
            log_prob: stacked log_prob in shape (num_agent,)
            reward: use avarage reward of two actors in shape (1,) as the learning reward,
                as the project target is calculated by taking the maximum over both agents,
                it sets the lower bound as the avarage reward is always less than max reward.
            done: done flag of one actor in shape (1,), as the done flags are same for them
        '''
        is_full = self.memory.add(state, action, log_prob, reward, done, next_state)
        # the self.update_flag equates to 0 every UPDATE_PER_STEP time steps.
        self.update_flag = (self.update_flag + 1) % UPDATE_PER_STEP
        if is_full and self.update_flag == 0:
            # train the model for NUM_EPOCH times.
            for _ in range(NUM_EPOCH):
                experiences = self.memory.sample(self.actor_0, self.actor_1, self.critic_local, batch_size=BATCH_SIZE)
                self.learn(experiences)

    def learn(self, experience):
        '''
        The function perform HAPPO algorithm, the details are in ./Report.md

        Params:
            experience: A mini batch of shuffled experiences containing states, actions, values, log_probs and advantages
        '''
        states, actions, values, log_probs, advantages = experience

        # initialize the weighted_ratio for actors sequence update, the details are in ./Report.md
        weighted_ratio = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actors_id = [0, 1] if random.random() > 0.5 else [1, 0]
        for i in actors_id:
            actor = getattr(self, f'actor_{i}')
            # calculate the new distribution to update model parameters by backpropagation.
            dist = actor(states[:, i, :].squeeze(), std=STD)
            new_log_probs = dist.log_prob(actions[:, i, :].squeeze())

            # calculate the HAPPO loss for each actor, the details are in ./Report.md
            factor = (new_log_probs - log_probs[:, i].squeeze()).exp()
            weighted_ratio = weighted_ratio * factor
            weighted_clipped_ratio = torch.clamp(weighted_ratio, (1 - CLIP_FACTOR) * weighted_ratio, (1 + CLIP_FACTOR) * weighted_ratio)
            ppo_loss = -torch.min(weighted_ratio, weighted_clipped_ratio).mean()

            # update model parameters, apply learning rate decay
            opt = getattr(self, f'actor_{i}_opt')
            sch = getattr(self, f'actor_{i}_scheduler')
            opt.zero_grad()
            ppo_loss.backward()
            opt.step()
            sch.step()

            # detach the new weighted_ratio for next actor
            weighted_ratio = weighted_ratio.detach()
        
        # calculate value net loss
        returns = advantages + values
        c_value = self.critic_local(states.flatten(-2, -1)).squeeze()
        v_loss = F.mse_loss(c_value, returns)
        
        # backpropagation
        
        self.critic_opt.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_opt.step()
        self.critic_scheduler.step()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, scale=0.1, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.

        :param size: Integer. Dimension of each state
        :param seed: Integer. Random seed
        :param scale: Float. Scale of the distribution
        :param mu: Float. Mean of the distribution
        :param theta: Float. Rate of the mean reversion of the distribution
        :param sigma: Float. Volatility of the distribution
        """
        self.size = size
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()
        random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.size) * self.mu

    def sample(self, scale=None, sigma=None):
        """Update internal state and return it as a noise sample."""
        scale = scale or self.scale
        sigma = sigma or self.sigma
        x = self.state
        dx = self.theta * (self.mu - x) + sigma * np.random.randn(self.size)
        self.state = x + dx
        gain = self.state * scale
        # print(gain, end='\r')
        return torch.from_numpy(gain).float().to(DEVICE)
    

if __name__ == '__main__':
    ...
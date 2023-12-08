[//]: # (Image References)


[image1]: image/advantage_decomposition.png "Advantage Decomposition"
[image2]: image/sequential_update.png "Sequential Update"
[image3]: image/psudocode.png "Psudocode"
[image4]: image/training_curve.png "Learning Curve"

***

# 1. Learning Algorithm
The algorithm is based on [HAPPO(Heterogeneous-Agent Proximal Policy Optimisation)](https://arxiv.org/abs/2109.11251).

HAPPO applies PPO algorithm on multi-agent environment with heterogeneous method. In the origin PPO algorithm, agents, even in cooperative games, could have conflicting directions of
policy updates. However, HAPPO do not need agents to share parameters, nor do they need any restrictive assumptions on decomposibility of the joint value function.

## PPO algorithm

[PPO algorithm](https://arxiv.org/abs/1707.06347) is a policy-based reinforcement learning algorithm with simplicity, stability, and sample efficiency. 

PPO was developed by John Schulman of OpenAI in 2017, it waw derived from Trust Region Policy Optimization (TRPO).

TRPO maximizes a “surrogate” objective:

 ${\text{CPI}}\left({\theta}\right) = \hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t}\mid{s_{t}}\right)}{\pi_{\theta_{old}}\left(a_{t}\mid{s_{t}}\right)})\hat{A}_{t}\right] = \hat{\mathbb{E}}_{t}\left[r_{t}\left(\theta\right)\hat{A}_{t}\right]$
 
 The probability ratio, $\frac{\pi_{\theta}\left(a_{t}\mid{s_{t}}\right)}{\pi_{\theta_{old}}\left(a_{t}\mid{s_{t}}\right)}$, may lead to an excessively large policy update and go far away from the optimal parameters.

PPO introduces the clip ratio method to limit the spread of the old distribution and new distribution. The objective of PPO is：

 ${\text{CLIP}}\left({\theta}\right) = \hat{\mathbb{E}}_{t}\left[\min\left(r_{t}\left(\theta\right)\hat{A}_{t}, \text{clip}\left(r_{t}\left(\theta\right), 1-\epsilon, 1+\epsilon\right)\hat{A}_{t}\right)\right]$

With a hyperparameter $\epsilon$, PPO limits the change of the old and new policy distribution, make sure the update is stable. 

## HAPPO algorithm
The **Heterogeneous-Agent Proximal Policy Optimization (HAPPO)** algorithm is based on TRPO and PPO algorith for multi-agent reinforcement learning. It does not require agents to share parameters, nor does it need any restrictive assumptions on decomposability of the joint value function. 

The algorithm is based on the **sequential update scheme** with theoretically-justified monotonic improvement guarantee.

It applies the **Multi-Agent Advantage Decomposition** method to update actors sequentially:

### Multi-Agent Advantage Decomposition

In any cooperative Markov games, given a
joint policy π, for any state s, and any agent subset i1:m, the below equations holds.

$A_\pi^{i_{1:m}}(s, a^{i_1:m}) = \displaystyle\sum_{j=1}^mA_\pi^{i_j}(s, \text{a}^{i_{1:j-1}}, a^{i_j})$

### Maximize the Expected Multi-Agent Advantage

HAPPO update actors' parameters sequentially based on advantage decomposition. 

The learning target for the actor is to maximize the expected multi-agent advantage, which is estimated as follows:

$\mathbb{E}_{a^{i_{1:m-1}}\sim\pi_{\theta_{k+1}}^{i_{1:m-1}}, a_{i_m}\sim\pi_{\theta^{i_m}}^{i_m}}[A_{\pi_{\theta_k}}(s, {\text{a}}^{i_{1:m-1}}, a^{i_m})]$

Which can be decomposed as follows:

![Sequential Update][image2]

Combine the idea of clipped importance sampling in PPO, we get a loss function as follows:

$\mathbb{E}_{s\sim\rho_{\pi_{\theta_k}}, a\sim\pi_{\theta_k}}[min(\frac{\pi_{\theta^{i_m}}^{i_m}(a^i|s)}{\pi_{\theta_{k}^{i_m}}^{i_m}(a^i|s)}M^{i_{1:m}(s, \text{a})}, clip(\frac{\pi_{\theta^{i_m}}^{i_m}(a^i|s)}{\pi_{\theta_{k}^{i_m}}^{i_m}(a^i|s)}, 1\pm\epsilon)M^{i_{1:m}(s, \text{a})})]$

HAPPO pemutates the order of actor updating in each learning epoch, and update each actor sequentially, based on the multi agent advantage decomposition method.

The monotonic improving property is given by:

$J(\bar\pi) \ge J(\pi) + \displaystyle\sum_{m=1}^n[L_\pi^{i_{1:m}}(\bar\pi^{i_{1:m-1}}, \bar\pi^{i_m}) - CD_{KL}^{max}(\pi^{i_m}, \bar\pi^{i_m})]$

## Design

This HAPPO algorithm components are as follows:
1. Actor network
2. Critic network
3. Buffer
4. OU noise
5. Agent

The Psudocode presents how the algorithm functions:
![Psudocode][image3]


## Actor network

The actor network accepts the local observation, forward it with the deep neural network, generate and return a multivariate normal distribution for each actor.

### Network architecture

1. local observation in (..., 24) -> [Linear(24, 128)] -> hidden state in (..., 128)
2. hidden state in (..., 128) -> [Linear(128, 128), ReLU] * 2 -> hidden state in (..., 128)
3. hidden state in (..., 128) -> [Linear(128, 64), ReLU] -> hidden state in (..., 64)
4. hidden state in (..., 64) -> [Linear(64, 64), ReLU] -> hidden state in (..., 64)
5. hidden state in (..., 64) -> [Linear(64, 2), Tanh] -> mu in (..., 2)
6. generate MultivariateNormal distribution with the mean mu and the standard deviation sigma

It receives the local observation for each actor in shape (batch size, state size) and output a 2 dims tensor with value between -1 and 1 as the mean of action. Conbined with a mamual standard deviation $\sigma$ for the explorition in training, the more states would be explored with the bigger $\sigma$. The final output is a multivariate normal distribution. The action and log probalitiy is sampled from this distribution in training, and a deterministic action in evaluation/prediction. 

LinearLR learning scheduler is applied for learing rate decay.

### Hyperparameters for Actor

1. ACTOR_LR **(actor initial learning rate)** = 5e-6
2. start_factor  **(start factor for LinearLR learning rate scheduler)** = 1
3. end_factor **(end factor for LinearLR learning rate scheduler)** = 0.5
4. total_iters **(total interations for LinearLR learning rate scheduler)** = 40000
5. STD **(constant standard diviation for multivariate normal distribution)** = 0.01 for training, 1e-8 for evaluation


## Critic Network

The critic network accepts the global state, forward it with the deep neural network, return a (batch size, 1) estimated value for the state.

According to HAPPO algorithm, I maintaince only one global critic network to provide the joint advantage estimation.

Finally, critic network uses a MSE loss between predicted value and GAE estimated value to update parameters.

LinearLR learning scheduler is applied for learing rate decay.

### Hyperparameters for Critic

1. CRITIC_LR **(actor initial learning rate)** = 5e-5
2. start_factor  **(start factor for LinearLR learning rate scheduler)** = 1
3. end_factor **(end factor for LinearLR learning rate scheduler)** = 0.5
4. total_iters **(total interations for LinearLR learning rate scheduler)** = 40000

## Buffer
The buffer is used to collect trajectories for each episode. The state, action, value, log probability, reward and done flag will be stored into buffer in each time step.

The buffer uses a fixed-length **(BUFFER_SIZE)** numpy ndarray to collect trajectories, calculates the [GAE(Generalized Advantage Estimation)](https://arxiv.org/abs/1506.02438) values, advantages, and offers the sampled batch with **BATCH_SIZE** by random choice.

The experiences stored in the buffer are in shape: (time step, num_agent, *data shape). 

The sampled batch experience are in shape: (batch size, *data shape).

### GAE calculation

To reduce the viarance of training, we introduce the [GAE advantage](https://arxiv.org/abs/1506.02438). 

The GAE advantages are calculated with the following formula with the reversed value sequence:

$\delta_t = r_t + \gamma{V}(s_{t+1}) - {V}(s_t)$

$\hat{A_t} = \delta_t + \gamma\delta_{t+1} + ... + (\gamma\delta)^{T-t+1}\delta_{T-1}$

$\hat{A_t} = \delta_t + \gamma\lambda\hat{A_{t+1}}$

$\lambda$ adjusts the bias-variance tradeoff. I find a lower $\lambda$ and shorter trajectory may improve learning performance in this project.

The GAE advantages are calculated when sampling. It will be calculated as follows:

1. extract the values, rewards, done flags sequence from the buffer
2. normalize the rewards to reduce training variance
3. calculate the next state value with target critic network to the value sequence
4. initiate a (time step, num_agent) zero ndarray to contain advantages
5. initiate a (num_agent) dim zero ndarray to contain advantages for each time step
6. use the previous formula to calculate advantages recursively. 

### Sampling
To sample a *BATCH_SIZE* samples, the **sample** function calculates the current values, log probabilities and advantages with the experiences sequence, and chooses a batch size of experiences randomly.

### Hyperparameters for Buffer
1. BUFFER_SIZE **(the max time steps experiences sequences stored in a rollout buffer)**: 2048
2. BATCH_SIZE **(the batch size to be sample for each learning step)**: 128
3. GAMMA **(The gamma decay factor to discount the reward or value)**: 0.99
4. GAE_LAMBDA **(the gae_lambda factor to discount the next step GAE advantage)**: 0.95


## Agent
The agent orgnizes actor network, local critic network, target critic network and buffer together, provides the **step** and **learn** functions to train the model.

It plays two important roles:
1. Interact with the environment to generate action and log probability for experiences collection
2. Sample previous experiences and use HAPPO algorithm to update current actor and critic parameters for optimization, in every UPDATE_PER_STEP time steps.

### Hyperparameters for Buffer
1. NUM_EPOCH **(how many times the updating taking place if the learning conditions are matched)**: 8
2. UPDATE_PER_STEP **(how many time steps to trigger learning process)**: 128

***

# 2. Plot of Rewards

The training curve, with a big actor network $\sigma$:

![Learning Curve][image4]


***

# 3. Ideas for Future Work

1. Improve the efficiency of hyper parameters tuning
2. Apply RNN or transformer networks in the model
3. Improve value loss function
4. Find more efficient way to balance exploration and exploitation. Apply gSDE method.
5. Try more advanced algorithms.
6. Use more efficient sampling and learning method to achieve the target in fewer episodes.
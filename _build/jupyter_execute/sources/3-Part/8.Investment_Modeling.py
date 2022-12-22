#!/usr/bin/env python
# coding: utf-8

# # 8. Investment Modeling

#   
# 
# ## 8.1 Introduction to investment modeling
# 

# 
# Investment modeling is a type of financial modeling that aims to forecast the performance of a financial asset or portfolio of assets. It is commonly used by investors, financial analysts, and portfolio managers to make informed decisions about buying, selling, or holding financial assets. There are many different techniques and approaches to investment modeling, including fundamental analysis, technical analysis, and quantitative analysis.
# 
# Fundamental analysis involves examining the underlying economic and financial factors that may influence the value of an asset. This may include analyzing the financial statements and performance of a company, as well as macroeconomic factors such as interest rates and economic growth. Technical analysis involves using statistical and graphical techniques to analyze the historical price and volume data of a financial asset in order to forecast its future performance. Quantitative analysis involves using mathematical and statistical techniques to analyze data and make predictions about future outcomes.
# 
# Investment modeling can be used to forecast the performance of a single asset, such as a stock or bond, or a portfolio of assets, such as a mutual fund or exchange-traded fund (ETF). It can also be used to optimize portfolio construction and risk management strategies.
# 

# ## 8.2 Portfolio optimization

# In investment modeling, portfolio optimization refers to the process of selecting a set of investments that maximizes returns while minimizing risk. This can be achieved by finding the optimal allocation of assets within a portfolio, taking into account factors such as the expected returns, risks, and correlations of the different assets.
# 
# One common approach to portfolio optimization is to use mean-variance optimization, where the portfolio is constructed to maximize the expected return for a given level of risk, or to minimize risk for a given level of expected return. This involves calculating the mean and variance of the returns for each asset in the portfolio, and using optimization algorithms to find the optimal weightings for the assets.
# 
# Another approach to portfolio optimization is to use risk-parity, where the portfolio is constructed such that each asset contributes equally to the overall risk of the portfolio. This can be achieved by setting the weights of the assets inversely proportional to their volatility, such that less volatile assets have higher weights and more volatile assets have lower weights.
# 
# In both cases, the optimization process typically involves constraints on the weights of the assets, such as minimum and maximum weightings and limits on the total number of assets in the portfolio.
# 
# In the context of deep learning, portfolio optimization can be tackled using techniques such as reinforcement learning, where a model is trained to maximize returns while minimizing risk by learning from historical data and making decisions based on the predicted outcomes.

# ### example
# 
# Here is an example of using reinforcement learning for portfolio optimization in Python using the PyTorch library:

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the reinforcement learning agent
class Agent(nn.Module):
    def __init__(self, input_size, output_size):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the agent and set the learning rate
agent = Agent(input_size=n_stocks, output_size=n_stocks)
learning_rate = 0.001
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

# Set the number of episodes and the discount factor
n_episodes = 1000
discount_factor = 0.99

# Train the agent using reinforcement learning
for episode in range(n_episodes):
    # Get the current state and calculate the action probabilities
    state = torch.tensor(portfolio_weights, dtype=torch.float).unsqueeze(0)
    action_probs = agent(state)

    # Sample an action from the action probabilities
    action = torch.multinomial(action_probs, 1).item()

    # Calculate the reward and the next state
    reward = portfolio_returns[action]
    next_state = torch.tensor(new_weights[action], dtype=torch.float).unsqueeze(0)

    # Calculate the expected return
    expected_return = reward + discount_factor * agent(next_state).max().item()

    # Calculate the loss and optimize the model
    loss = -torch.log(action_probs[0, action]) * expected_return
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training, use the trained model to select the optimal portfolio weights
state = torch.tensor(portfolio_weights, dtype=torch.float).unsqueeze(0)
action_probs = agent(state)
optimal_action = action_probs.argmax().item()
optimal_weights = new_weights[optimal_action]


# This code defines a reinforcement learning agent with a simple fully-connected neural network and trains it to optimize the portfolio weights by maximizing the expected return. The agent takes the current portfolio weights as input and outputs a probability distribution over the possible actions (i.e., the possible new portfolio weights). The agent samples an action from this probability distribution, calculates the reward (i.e., the return of the new portfolio), and updates the model based on the expected return. 
# 
# Once the portfolio has been updated, the process is repeated for the next time step. This continues until the end of the investment horizon is reached. At that point, the final portfolio and its associated returns are calculated and compared to the benchmark.
# 
# One key aspect of reinforcement learning in investment modeling is the choice of reward function. This function determines the goal of the agent and guides its actions. In the case of portfolio optimization, a common choice is to maximize the portfolio's Sharpe ratio, which measures the risk-adjusted returns of the portfolio. Other reward functions, such as the Sortino ratio or the Omega ratio, can also be used.
# 
# It's important to note that reinforcement learning in investment modeling is still an active area of research, and there are many open questions and challenges that need to be addressed. Some of these include how to effectively handle non-stationary environments, how to incorporate constraints on the portfolio, and how to handle large-scale problems with high dimensional state and action spaces. Despite these challenges, the potential for reinforcement learning to improve investment decision-making makes it an exciting area to watch in the future.

# ### example with stable-baselines
# 
# Here is an example of portfolio optimization using reinforcement learning in Python with the library **`stable-baselines3`**:

# In[ ]:


import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ddpg.policies import MlpPolicy

# Load the data
data = pd.read_csv('stock_prices.csv')

# Preprocess the data
returns = data.pct_change().dropna()

# Define the environment
n_assets = returns.shape[1]

def get_obs():
  return np.expand_dims(returns.values[-1], axis=0)

def reward_fun(action):
  action = np.squeeze(action)
  portfolio_return = np.sum(returns.values[-1] * action)
  sharpe_ratio = portfolio_return / np.std(returns.values[-1] * action)
  return sharpe_ratio

env = DummyVecEnv([lambda: returns])

# Create the reinforcement learning agent
model = DDPG(MlpPolicy, env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Test the agent
action, _states = model.predict(get_obs())
print(action)


# ## 8.3 Risk assessment
# 
# 

# Risk assessment in investment modeling refers to the process of evaluating and measuring the potential risks associated with an investment. This is an important step in the investment process as it helps investors make informed decisions about the allocation of their funds.
# 
# There are several approaches to risk assessment in investment modeling, including statistical methods, fundamental analysis, and technical analysis. Statistical methods involve the use of statistical tools and techniques to evaluate the risk associated with an investment. Fundamental analysis involves evaluating the financial health and performance of a company, while technical analysis involves studying the historical price and volume data of a security.
# 
# In addition to these approaches, machine learning techniques are also increasingly being used for risk assessment in investment modeling. Machine learning algorithms can analyze large amounts of data and identify patterns and trends that may be indicative of potential risks. For example, a machine learning model trained on historical stock market data could be used to identify patterns that may be indicative of a market crash.
# 
# Overall, risk assessment is an important step in the investment process as it helps investors make informed decisions about the allocation of their funds. By evaluating the potential risks associated with an investment, investors can make more informed decisions about the allocation of their funds and potentially minimize their risk of losses.
# 

# 
# ## 8.4 Case study: Algorithmic trading
# 

# 
# In algorithmic trading, AI models are used to analyze market data and make trades based on predetermined rules. These models can be trained to make trades in response to certain patterns or signals in the market data, such as a sudden change in the price of a particular asset.
# 
# One example of using AI for algorithmic trading is using a reinforcement learning model to optimize a portfolio. In this case, the model would receive a reward for making profitable trades and a penalty for making unprofitable ones. The model would then learn to make trades that maximize the reward over time.
# 
# To implement this type of model, we would first need to gather market data and pre-process it to extract relevant features. We could then use this data to train a reinforcement learning model using a technique such as Q-learning. Once the model is trained, we can use it to make trades in real-time as new market data becomes available.
# 
# Here is an example of how to implement a Q-learning model for portfolio optimization in Python using the PyTorch library:

# In[ ]:


import torch
import numpy as np

# Define the Q-learning model
class QLearner(torch.nn.Module):
  def __init__(self, num_assets, learning_rate=0.01):
    super().__init__()
    self.num_assets = num_assets
    self.learning_rate = learning_rate

    # Define the layers of the model
    self.fc1 = torch.nn.Linear(num_assets, 64)
    self.fc2 = torch.nn.Linear(64, 32)
    self.fc3 = torch.nn.Linear(32, 1)

  def forward(self, x):
    # Perform forward pass through the model
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# Initialize the model
model = QLearner(num_assets=10)

# Define the optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate)
loss_fn = torch.nn.MSELoss()

# Loop through training iterations
for i in range(1000):
  # Sample a batch of data and labels
  data = torch.tensor(np.random.randn(32, 10), dtype=torch.float)
  labels = torch.tensor(np.random.randn(32, 1), dtype=torch.float)

  # Zero the gradients
  optimizer.zero_grad()

  # Perform a forward pass through the model
  outputs = model(data)

  # Calculate the loss
  loss = loss_fn(outputs, labels)

  # Backpropagate the gradients
  loss.backward()

  # Update the model weights
  optimizer.step()

# Use the trained model to make trades
trades = model(market_data)


# 

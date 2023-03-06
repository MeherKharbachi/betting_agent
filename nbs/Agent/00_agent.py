#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#| default_exp Agent.agent


# In[ ]:


#| hide
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
from IPython.core.debugger import set_trace


# # D3rlpy Agent
# 
# >  Simulate a trading strategy using a custom football betting environment. 

# In[ ]:


#| export

import pandas as pd
import d3rlpy
import torch
from betting_agent.Utils.uncache import *
from betting_agent.Utils.monkey_patching import *
from d3rlpy.preprocessing.scalers import Scaler
from betting_agent.config.localconfig import CONFIG, DB_HOSTS
from betting_env.betting_env import BettingEnv
from betting_env.utils.data_extractor import *


# ## Load Data

# In[ ]:


fixtures = data_aggregator(
    db_hosts=DB_HOSTS, config=CONFIG, db_host="prod_atlas", limit=4
)
fixtures.head()


# ### Apply Monkey-patching

# In[ ]:


#| export

from d3rlpy import torch_utility
from d3rlpy.online.buffers import ReplayBuffer


# In[ ]:


#| export
torch_utility.torch_api = torch_api
ReplayBuffer.append = append
ReplayBuffer._add_last_step = add_last_step
uncache(["d3rlpy.torch_utility","d3rlpy.online.buffers"])


# ## D3rlpy Agent

# In[ ]:


#| export

from betting_agent.Utils.scaler import CustomScaler
from betting_agent.Utils.network_architecture import *
from d3rlpy.algos import DQN
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.preprocessing.scalers import register_scaler
from torch.optim import Adam


# We propose a function that will prepare the `Reinforcement learning` algorithm prior to training. Initially, we initialise the `Betting environment` with the supplied data, then we set up the `Scaler`, which will transform our observations to particular features from the Database, and last, we set up the `Buffer`; `D3rlpy` supports both offline and online training tools. In this case, the `Buffer` will try several experiences in order to obtain a useful dataset.
# 
# Furthemore, we supply additionally an `Optimizer` to update weights and reduce losses for the `Neural Network` and an `Explorer` which will apply the `exploration-exploitation` dilemma which must exist side by side because The majority of the time, the `epsilon-greedy` strategy takes the action with the largest estimated reward. `Exploration` allows us to experiment with new ideas, which are frequently at contradiction with what we have already learned. The procedure starts with 100% `exploration` and subsequently decreases to 10%.
# 
# We should note that the `D3rlpy` package has several `RL` algorithms; in our situation, we will choose the `DQN` algorithm (Deep Q-Network).

# In[ ]:


#| export


def rl_algo_preparation(
    fixtures: pd.DataFrame,  # All provided games.
    algo: d3rlpy.algos = DQN,  # D3rlpy RL algorithm.
    algo_batch_size=32,  #  Mini-batch size.
    algo_learning_rate=2.5e-4,  # Algo learning rate.
    algo_target_update_interval=100,  # Interval to update the target network.
    algo_scaler: Scaler = CustomScaler,  # The scaler for data transformation.
    optimizer: torch.optim = Adam,  # Algo Optimizer.
    optimizer_weight_decay=1e-4,  # Optimizer weight decay.
    maxlen_buffer=1000000,  #  The maximum number of data length.
    explorer_start_epsilon=1.0,  # The beginning epsilon.
    explorer_end_epsilon=0.1,  # The end epsilon.
    explorer_duration=100000,  # The scheduling duration.
):
    "Prepare RL algorithm components."
    # Init betting env.
    env = BettingEnv(fixtures)

    # Init Scaler.
    register_scaler(algo_scaler)
    custom_scaler = algo_scaler()

    # Init Buffer.
    buffer = ReplayBuffer(env=env, maxlen=maxlen_buffer)

    # Init the epsilon-greedy explorer
    explorer = LinearDecayEpsilonGreedy(
        start_epsilon=explorer_start_epsilon,
        end_epsilon=explorer_end_epsilon,
        duration=explorer_duration,
    )

    # Init Optimizer.
    optim_factory = OptimizerFactory(optimizer, weight_decay=optimizer_weight_decay)

    # Init RL Algo.
    rl_algo = algo(
        batch_size=algo_batch_size,
        learning_rate=algo_learning_rate,
        target_update_interval=algo_target_update_interval,
        optim_factory=optim_factory,
        scaler=custom_scaler,
        encoder_factory=CustomEncoderFactory(feature_size=env.action_space.n),
    )

    return env, buffer, explorer, rl_algo


# In[ ]:


#| export

from d3rlpy.algos.base import AlgoBase


# In[ ]:


#| export

AlgoBase.fit_online = fit_online
uncache(["d3rlpy.torch_utility", "d3rlpy.online.buffers", "d3rlpy.algos.base"])


# Launch training
# 

# In[ ]:


#| export

def launch_training(
    fixtures: pd.DataFrame,  # All provided games.
    training_steps: int = 100,  # The number of total steps to train.
    n_steps_per_epoch: int = 50,  # The number of steps per epoch.
    update_start_step: int = 50,  #  The steps before starting updates.
    algo: d3rlpy.algos = DQN,  # D3rlpy RL algorithm.
    algo_batch_size=32,  #  Mini-batch size.
    algo_learning_rate=2.5e-4,  # Algo learning rate.
    algo_target_update_interval=100,  # Interval to update the target network.
    algo_scaler: Scaler = CustomScaler,  # The scaler for data transformation.
    optimizer: torch.optim = Adam,  # Algo Optimizer.
    optimizer_weight_decay=1e-4,  # Optimizer weight decay.
    maxlen_buffer=1000000,  #  The maximum number of data length.
    explorer_start_epsilon=1.0,  # The beginning epsilon.
    explorer_end_epsilon=0.1,  # The end epsilon.
    explorer_duration=100,  # The scheduling duration.
    show_progress: bool = True,  # Flag to show progress bar for iterations.
    save_metrics: bool = True,  # Flag to record metrics. If False, the log directory is not created and the model parameters are not saved.
):
    "Launch RL algorithm training."
    # Get algo params.
    env, buffer, explorer, rl_algo = rl_algo_preparation(
        fixtures=fixtures,
        algo=algo,
        algo_batch_size=algo_batch_size,
        algo_learning_rate=algo_learning_rate,
        algo_target_update_interval=algo_target_update_interval,
        algo_scaler=algo_scaler,
        optimizer=optimizer,
        optimizer_weight_decay=optimizer_weight_decay,
        maxlen_buffer=maxlen_buffer,
        explorer_start_epsilon=explorer_start_epsilon,
        explorer_end_epsilon=explorer_end_epsilon,
        explorer_duration=explorer_duration,
    )
    # Launch training.
    rl_algo.fit_online(
        env,  # Gym environment.
        buffer,  # Buffer.
        explorer,  # Explorer.
        n_steps=training_steps,  # Train for 'training_steps' steps.
        n_steps_per_epoch=n_steps_per_epoch,  # Evaluation is performed every 'n_steps_per_epoch' steps.
        update_start_step=update_start_step,  # Parameter update starts after 'update_start_step' steps.
        save_metrics=save_metrics,  # Save metrics.
        show_progress=show_progress,  # Show progress.
    )


# In[ ]:


launch_training(
    fixtures=fixtures,
    algo=DQN,
    algo_scaler=CustomScaler,
    optimizer=Adam,
    explorer_duration=1000,
    training_steps=1000,
    n_steps_per_epoch=50,  
    update_start_step=50,
    save_metrics=True
)


# In[ ]:


#| hide
from nbdev import nbdev_export

nbdev_export()


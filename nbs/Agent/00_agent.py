#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#| default_exp Agent.agent


# In[ ]:


#| hide

from IPython.core.debugger import set_trace

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# # D3rlpy Agent
# 
# >  Simulate a trading strategy using a custom football betting environment. 

# In[ ]:


#| export
import pandas as pd
from betting_agent.config.localconfig import CONFIG, DB_HOSTS
from betting_env.betting_env import BettingEnv
from betting_env.utils.data_extractor import *
from typing_extensions import Protocol
from d3rlpy.dataset import TransitionMiniBatch
from betting_agent.Utils.uncache import *
from betting_agent.Utils.monkey_patching import *


# ## Load Data

# In[ ]:


fixtures = data_aggregator(
    db_hosts=DB_HOSTS, config=CONFIG, db_host="prod_atlas", limit=4
)
fixtures.head()


# ## D3rlpy Agent

# Load the betting environment

# In[ ]:


#| export
env = BettingEnv(fixtures)


# ### Apply Monkey-patching

# In[ ]:


# | export

from d3rlpy import torch_utility
from d3rlpy.online.buffers import ReplayBuffer


# In[ ]:


#| export
torch_utility.torch_api = torch_api
ReplayBuffer.append = append
ReplayBuffer._add_last_step = add_last_step
uncache(["d3rlpy.torch_utility","d3rlpy.online.buffers"])


# ### Train Agent

# In[ ]:


#| export

from betting_agent.Utils.scaler import CustomScaler
from betting_agent.Utils.network_architecture import *
from d3rlpy.algos import DQN
from torch.optim import Adam
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.preprocessing.scalers import register_scaler


# Init our custom transformer

# In[ ]:


register_scaler(CustomScaler)


# D3rlpy provides not only offline training, but also online training utilities. Here the buffer will try different experiences to collect a decent dataset.

# In[ ]:


buffer = ReplayBuffer(maxlen=1000000, env=env)


# The majority of the time, the epsilon-greedy strategy chooses the action with the highest estimated reward. Exploration and exploitation should coexist in harmony. Exploration gives us the freedom to experiment with new ideas, often at contradiction with what we have already learnt.
# 
# The process begins with 100% exploration and gradually reduces to 10%.

# In[ ]:


# Init the epsilon-greedy explorer
explorer = LinearDecayEpsilonGreedy(start_epsilon=1.0,
                                    end_epsilon=0.1,
                                    duration=100000)


# We init an Optimizer to update weights and reduce losses for the Neural Network
# 

# In[ ]:


optim_factory = OptimizerFactory(Adam, weight_decay=1e-4)


# Next, we select an RL technique to train our agent. In this example, we will use the fundamental approach 'DQN' (Deep Q-Network).

# In[ ]:


custom_scaler = CustomScaler()

dqn = DQN(
    batch_size=32,
    learning_rate=2.5e-4,
    target_update_interval=100,
    optim_factory=optim_factory,
    scaler=custom_scaler,
    encoder_factory=CustomEncoderFactory(feature_size=env.action_space.n)

)

dqn.build_with_env(env)


# In[ ]:


from d3rlpy.algos.base import AlgoBase


# In[ ]:


AlgoBase.fit_online = fit_online
uncache(["d3rlpy.torch_utility","d3rlpy.online.buffers","d3rlpy.algos.base"])


# Launch training
# 

# In[ ]:


dqn.fit_online(
    env,
    buffer,
    explorer,
    n_steps=10,  # train for 100K steps
    n_steps_per_epoch=5,  # evaluation is performed every 100 steps
    update_start_step=5,  # parameter update starts after 100 steps
    eval_epsilon=0.3,
    save_metrics=False,
    # tensorboard_dir= 'runs'
)


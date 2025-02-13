# -*- coding: utf-8 -*-
"""The main script to run the planner agent."""

import os

import agentscope
from agentscope.agents import UserAgent
from agentscope.service import ServiceToolkit
from examples.agent_planner.planner_agent import PlannerAgent


# Initialize agentscope
agentscope.init(
    model_configs={
        "config_name": "my_config",
        "model_type": "dashscope_chat",
        "model_name": "qwen-max",
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
    },
)

# Create the user and assistant agents
user = UserAgent(name="User")

toolkit = ServiceToolkit()
assistant = PlannerAgent(
    name="Friday",
    service_toolkit=toolkit,
    model_config_name="my_config",
    max_iters=50,
    verbose=True,
)

# Begin the conversation
msg = None
while True:
    msg = assistant(msg)
    msg = user(msg)
    if msg.content == "exit":
        break

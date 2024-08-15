# %% [markdown]
# # How to add a custom system prompt to the prebuilt ReAct agent
#
# This tutorial will show how to add a custom system prompt to the prebuilt ReAct agent. Please see [this tutorial](./create-react-agent.ipynb) for how to get started with the prebuilt ReAct agent
#
# You can add a custom system prompt by passing a string to the `state_modifier` param.

# %% [markdown]
# ## Setup

# %%
# %%capture --no-stderr
# %pip install -U langgraph langchain-openai

# %%
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")

# Recommended
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Create ReAct Agent Tutorial"

# %% [markdown]
# ## Code

# %%
# First we initialize the model we want to use.
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)


# For this tutorial we will use custom tool that returns pre-defined values for weather in two cities (NYC & SF)

from typing import Literal

from langchain_core.tools import tool


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

# We can add our system prompt here

prompt = "Respond in Italian"

# Define the graph

from langgraph.prebuilt import create_react_agent

graph = create_react_agent(model, tools=tools, state_modifier=prompt)


# %% [markdown]
# ## Usage
#

# %%
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# %%
inputs = {"messages": [("user", "What's the weather in NYC?")]}

print_stream(graph.stream(inputs, stream_mode="values"))

# %%

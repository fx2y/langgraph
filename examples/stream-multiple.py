# %% [markdown]
# # How to configure multiple streaming modes at the same time

# %% [markdown]
# This guide covers how to configure multiple streaming modes at the same time.

# %% [markdown]
# ## Setup

# %% [markdown]
# We'll be using a simple ReAct agent for this guide.

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

# %%
from typing import Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


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

model = ChatOpenAI(model_name="gpt-4o", temperature=0)
graph = create_react_agent(model, tools)

# %% [markdown]
# ## Stream multiple

# %%
inputs = {"messages": [("human", "what's the weather in sf")]}
async for event, chunk in graph.astream(inputs, stream_mode=["updates", "debug"]):
    print(f"Receiving new event of type: {event}...")
    print(chunk)
    print("\n\n")

# %%

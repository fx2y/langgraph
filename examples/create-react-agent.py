# %% [markdown]
# # How to use the prebuilt ReAct agent

# %% [markdown]
# In this how-to we'll create a simple [ReAct](https://arxiv.org/abs/2210.03629) agent app that can check the weather. The app consists of an agent (LLM) and tools. As we interact with the app, we will first call the agent (LLM) to decide if we should use tools. Then we will run a loop:  
#
# 1. If the agent said to take an action (i.e. call tool), we'll run the tools and pass the results back to the agent
# 2. If the agent did not ask to run tools, we will finish (respond to the user)
#
# <div class="admonition warning">
#     <p class="admonition-title">Prebuilt Agent</p>
#     <p>
# Please note that here will we use a prebuilt agent. One of the big benefits of LangGraph is that you can easily create your own agent architectures. So while it's fine to start here to build an agent quickly, we would strongly recommend learning how to build your own agent so that you can take full advantage of LangGraph.
#     </p>
# </div>   

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


# Define the graph

from langgraph.prebuilt import create_react_agent

graph = create_react_agent(model, tools=tools)

# %% [markdown]
# ## Usage
#
# First, let's visualize the graph we just created

# %%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))


# %%
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# %% [markdown]
# Let's run the app with an input that needs a tool call

# %%
inputs = {"messages": [("user", "what is the weather in sf")]}
print_stream(graph.stream(inputs, stream_mode="values"))

# %% [markdown]
# Now let's try a question that doesn't need tools

# %%
inputs = {"messages": [("user", "who built you?")]}
print_stream(graph.stream(inputs, stream_mode="values"))

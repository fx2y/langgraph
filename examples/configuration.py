# %% [markdown]
# # How to add runtime configuration to your graph
#
# Sometimes you want to be able to configure your agent when calling it. 
# Examples of this include configuring which LLM to use.
# Below we walk through an example of doing so.

# %% [markdown]
# ## Base
#
# First, let's create a very simple graph

# %%
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import END, StateGraph, START

model = ChatAnthropic(model_name="claude-2.1")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def _call_model(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("model", _call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

app = workflow.compile()

# %%
app.invoke({"messages": [HumanMessage(content="hi")]})

# %% [markdown]
# ## Configure the graph
#
# Great! Now let's suppose that we want to extend this example so the user is able to choose from multiple llms.
# We can easily do that by passing in a config.
# This config is meant to contain things are not part of the input (and therefore that we don't want to track as part of the state).

# %%
from langchain_openai import ChatOpenAI

openai_model = ChatOpenAI()

models = {
    "anthropic": model,
    "openai": openai_model,
}


def _call_model(state, config):
    m = models[config["configurable"].get("model", "anthropic")]
    response = m.invoke(state["messages"])
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("model", _call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

app = workflow.compile()

# %% [markdown]
# If we call it with no configuration, it will use the default as we defined it (Anthropic).

# %%
app.invoke({"messages": [HumanMessage(content="hi")]})

# %% [markdown]
# We can also call it with a config to get it to use a different model.

# %%
config = {"configurable": {"model": "openai"}}
app.invoke({"messages": [HumanMessage(content="hi")]}, config=config)

# %% [markdown]
# We can also adapt our graph to take in more configuration! Like a system message for example.

# %%
from langchain_core.messages import SystemMessage


def _call_model(state, config):
    m = models[config["configurable"].get("model", "anthropic")]
    messages = state["messages"]
    if "system_message" in config["configurable"]:
        messages = [
            SystemMessage(content=config["configurable"]["system_message"])
        ] + messages
    response = m.invoke(messages)
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("model", _call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

app = workflow.compile()

# %%
app.invoke({"messages": [HumanMessage(content="hi")]})

# %%
config = {"configurable": {"system_message": "respond in italian"}}
app.invoke({"messages": [HumanMessage(content="hi")]}, config=config)

# %%

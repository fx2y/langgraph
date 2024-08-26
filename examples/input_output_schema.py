# %% [markdown]
# # How to define input/output schema for your graph
#
# By default, `StateGraph` takes in a single schema and all nodes are expected to communicate with that schema. However, it is also possible to define explicit input and output schemas for a graph. This is helpful if you want to draw a distinction between input and output keys.
#
# In this notebook we'll walk through an example of this. At a high level, in order to do this you simply have to pass in `input=..., output=...` when defining the graph. Let's see an example below!

# %%
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    answer: str


def answer_node(state: InputState):
    return {"answer": "bye"}


graph = StateGraph(input=InputState, output=OutputState)
graph.add_node(answer_node)
graph.add_edge(START, "answer_node")
graph.add_edge("answer_node", END)
graph = graph.compile()

graph.invoke({"question": "hi"})

# %% [markdown]
# Notice that the output of invoke only includes the output schema.

# %%

# %% [markdown]
# # How to add node retry policies
#
# There are many use cases where you may wish for your node to have a custom retry policy, for example if you are calling an API, querying a database, or calling an LLM, etc. 
#
# In order to configure the retry policy, you have to pass the `retry` parameter to the `add_node` function. The `retry` parameter takes in a `RetryPolicy` named tuple object. Below we instantiate a `RetryPolicy` object with the default parameters:

# %%
from langgraph.pregel import RetryPolicy

RetryPolicy()

# %% [markdown]
# If you want more information on what each of the parameters does, be sure to read the [reference](https://langchain-ai.github.io/langgraph/reference/graphs/#retrypolicy).
#
# ## Passing a retry policy to a node
#
# Lastly, we can pass `RetryPolicy` objects when we call the `add_node` function. In the example below we pass two different retry policies to each of our nodes:

# %%
import operator
import sqlite3
from typing import Annotated, Sequence, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from langgraph.graph import END, StateGraph, START
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage

db = SQLDatabase.from_uri("sqlite:///:memory:")

model = ChatAnthropic(model_name="claude-2.1")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def query_database(state):
    query_result = db.run("SELECT * FROM Artist LIMIT 10;")
    return {"messages": [AIMessage(content=query_result)]}


def call_model(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node(
    "query_database",
    query_database,
    retry=RetryPolicy(retry_on=sqlite3.OperationalError),
)
workflow.add_node("model", call_model, retry=RetryPolicy(max_attempts=5))
workflow.add_edge(START, "model")
workflow.add_edge("model", "query_database")
workflow.add_edge("query_database", END)

app = workflow.compile()

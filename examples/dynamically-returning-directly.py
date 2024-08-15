# %% [markdown]
# # How to let agent return tool results directly
#
# A typical ReAct loop follows user -> assistant -> tool -> assistant ..., -> user. In some cases, you don't need to call the LLM after the tool completes, the user can view the results directly themselves.
#
# In this example we will build a conversational ReAct agent where the LLM can optionally decide to return the result of a tool call as the final answer. This is useful in cases where you have tools that can sometimes generate responses that are acceptable as final answers, and you want to use the LLM to determine when that is the case

# %% [markdown]
# ## Setup
#
# First we need to install the packages required

# %%
# %%capture --no-stderr
# %pip install --quiet -U langgraph langchain_community langchain_openai tavily-python

# %% [markdown]
# Next, we need to set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)

# %%
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")

# %% [markdown]
# Optionally, we can set API key for [LangSmith tracing](https://smith.langchain.com/), which will give us best-in-class observability.

# %%
os.environ["LANGCHAIN_TRACING_V2"] = "true"
_set_env("LANGCHAIN_API_KEY")

# %% [markdown]
# ## Set up the tools
#
# We will first define the tools we want to use.
# For this simple example, we will use a built-in search tool via Tavily.
# However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/docs/modules/agents/tools/custom_tools) on how to do that.
#
# :::tip
#     We overwrite the default schema of the search tool to have **an additional parameter** for returning directly.
#     This extra argument isn't used by the tool, but our workflow will check for its value to determine how to route
#     the tool results.
# :::

# %%
from langchain_core.pydantic_v1 import BaseModel, Field


class SearchTool(BaseModel):
    """Look up things online, optionally returning directly"""

    query: str = Field(description="query to look up online")
    return_direct: bool = Field(
        description="Whether or the result of this should be returned directly to the user without you seeing what it is",
        default=False,
    )


# %%
from langchain_community.tools.tavily_search import TavilySearchResults

search_tool = TavilySearchResults(max_results=1, args_schema=SearchTool)
tools = [search_tool]

# %% [markdown]
# We can now wrap these tools in a simple ToolExecutor.
# This is a real simple class that takes in a ToolInvocation and calls that tool, returning the output.
# A ToolInvocation is any class with `tool` and `tool_input` attribute.
#

# %%
from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(tools)

# %% [markdown]
# ## Set up the model
#
# Now we need to load the chat model we want to use.
# Importantly, this should satisfy two criteria:
#
# 1. It should work with messages. We will represent all agent state in the form of messages, so it needs to be able to work well with them.
# 2. It should work with OpenAI function calling. This means it should either be an OpenAI model or a model that exposes a similar interface.
#
# Note: these model requirements are not requirements for using LangGraph - they are just requirements for this one example.
#

# %%
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)

# %% [markdown]
#
# After we've done this, we should make sure the model knows that it has these tools available to call.
# We can do this by converting the LangChain tools into the format for OpenAI function calling, and then bind them to the model class.
#

# %%
model = model.bind_tools(tools)

# %% [markdown]
# ## Define the agent state
#
# The main type of graph in `langgraph` is the [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph).
#
# This graph is parameterized by a state object that it passes around to each node.
# Each node then returns operations to update that state.
# These operations can either SET specific attributes on the state (e.g. overwrite the existing values) or ADD to the existing attribute.
# Whether to set or add is denoted by annotating the state object you construct the graph with.
#
# For this example, the state we will track will just be a list of messages.
# We want each node to just add messages to that list.
# Therefore, we will use a `TypedDict` with one key (`messages`) and annotate it so that the `messages` attribute is always added to.
#

# %%
import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# %% [markdown]
# ## Define the nodes
#
# We now need to define a few different nodes in our graph.
# In `langgraph`, a node can be either a function or a [runnable](https://python.langchain.com/docs/expression_language/).
# There are two main nodes we need for this:
#
# 1. The agent: responsible for deciding what (if any) actions to take.
# 2. A function to invoke tools: if the agent decides to take an action, this node will then execute that action.
#
# We will also need to define some edges.
# Some of these edges may be conditional.
# The reason they are conditional is that based on the output of a node, one of several paths may be taken.
# The path that is taken is not known until that node is run (the LLM decides).
#
# 1. Conditional Edge: after the agent is called, we should either:
#    a. If the agent said to take an action, then the function to invoke tools should be called
#    b. If the agent said that it was finished, then it should finish
# 2. Normal Edge: after the tools are invoked, it should always go back to the agent to decide what to do next
#
# Let's define the nodes, as well as a function to decide how what conditional edge to take.
#

# %%
from langchain_core.messages import ToolMessage

from langgraph.prebuilt import ToolInvocation


# %% [markdown]
# **MODIFICATION**
#
# We change the `should_continue` function to check whether return_direct was set to True

# %%
# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we check if it's suppose to return direct
    else:
        arguments = last_message.tool_calls[0]["args"]
        if arguments.get("return_direct", False):
            return "final"
        else:
            return "continue"


# %%
# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# %% [markdown]
# **MODIFICATION**
#
# We change the tool calling to get rid of the `return_direct` parameter (not used in the actual tool call)

# %%
# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    arguments = tool_call["args"]
    if tool_name == "tavily_search_results_json":
        if "return_direct" in arguments:
            del arguments["return_direct"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=arguments,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a ToolMessage
    tool_message = ToolMessage(
        content=str(response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}


# %% [markdown]
# ## Define the graph
#
# We can now put it all together and define the graph!
#
# **MODIFICATION**
#
# We add a separate node for any tool call where `return_direct=True`. The reason this is needed is that after this node we want to end, while after other tool calls we want to go back to the LLM. 

# %%
from langgraph.graph import END, StateGraph, START

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)

# Note the "action" and "final" nodes are identical!
workflow.add_node("action", call_tool)
workflow.add_node("final", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Final call
        "final": "final",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")
workflow.add_edge("final", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

# %%
from IPython.display import Image, display

display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# %% [markdown]
# ## Use it!
#
# We can now use it!
# This now exposes the [same interface](https://python.langchain.com/docs/expression_language/) as all other LangChain runnables.

# %%
from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")

# %%
from langchain_core.messages import HumanMessage

inputs = {
    "messages": [
        HumanMessage(
            content="what is the weather in sf? return this result directly by setting return_direct = True"
        )
    ]
}
for output in app.stream(inputs, stream_mode="values"):
    # stream() yields dictionaries with output keyed by node name
    for message in output["messages"]:
        message.pretty_print()
    print("\n---\n")

# %%

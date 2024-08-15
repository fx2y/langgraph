# %% [markdown]
# # Chat Agent Executor using prebuilt Tool Node
#
#
# In this example we will build a ReAct Agent that uses tool calling and the prebuilt ToolNode.

# %% [markdown]
# ## Setup
#
# First we need to install the packages required

# %%
# %%capture --no-stderr
# %pip install --quiet -U langgraph langchain langchain_openai tavily-python

# %% [markdown]
# Next, we need to set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)

# %%
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key:")

# %% [markdown]
# Optionally, we can set API key for [LangSmith tracing](https://smith.langchain.com/), which will give us best-in-class observability.

# %%
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key:")

# %% [markdown]
# ## Set up the tools
#
# We will first define the tools we want to use.
# For this simple example, we will use a built-in search tool via Tavily.
# However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/v0.2/docs/how_to/custom_tools) on how to do that.
#
# **MODIFICATION**
#
# We don't need a ToolExecutor when using ToolNode.
#

# %%
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]

# %% [markdown]
# ## Set up the model
#
# Now we need to load the chat model we want to use.
# Importantly, this should satisfy two criteria:
#
# 1. It should work with messages. We will represent all agent state in the form of messages, so it needs to be able to work well with them.
# 2. It should work with tool calling. This means it should be a model that implements `.bind_tools()`.
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
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# %% [markdown]
# ## Define the nodes
#
# We now need to define a few different nodes in our graph.
# In `langgraph`, a node can be either a function or a [runnable](https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel).
# There are two main nodes we need for this:
#
# 1. The agent: responsible for deciding what (if any) actions to take.
# 2. **MODIFICATION** The prebuilt ToolNode, given the list of tools. This will take tool calls from the most recent AIMessage, execute them, and return the result as ToolMessages.
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
from langgraph.prebuilt import ToolNode


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
tool_node = ToolNode(tools)

# %% [markdown]
# ## Define the graph
#
# We can now put it all together and define the graph!

# %%
from langgraph.graph import END, StateGraph, START

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

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
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

# %% [markdown]
# ## Use it!
#
# We can now use it!
# This now exposes the [same interface](https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel) as all other LangChain runnables.

# %%
from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
app.invoke(inputs)

# %% [markdown]
# This may take a little bit - it's making a few calls behind the scenes.
# In order to start seeing some intermediate results as they happen, we can use streaming - see below for more information on that.
#
# ## Streaming
#
# LangGraph has support for several different types of streaming.
#
# ### Streaming Node Output
#
# One of the benefits of using LangGraph is that it is easy to stream output as it's produced by each node.
#

# %%
inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")

# %% [markdown]
# ### Streaming LLM Tokens
#
# You can also access the LLM tokens as they are produced by each node. 
# In this case only the "agent" node produces LLM tokens.
# In order for this to work properly, you must be using an LLM that supports streaming as well as have set it when constructing the LLM (e.g. `ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)`)
#

# %%
inputs = {"messages": [HumanMessage(content="what is the weather in sf?")]}

async for output in app.astream_log(inputs, include_types=["llm"]):
    # astream_log() yields the requested logs (here LLMs) in JSONPatch format
    for op in output.ops:
        if op["path"] == "/streamed_output/-":
            # this is the output from .stream()
            ...
        elif op["path"].startswith("/logs/") and op["path"].endswith(
            "/streamed_output/-"
        ):
            # because we chose to only include LLMs, these are LLM tokens
            print(op["value"])

# %%

# %% [markdown]
# # Respond in a format
#
# In this example we will build a chat executor that responds in a specific format. We will do this by using OpenAI function calling. This is useful when you want to enforce that an agent's response is in a specific format. In this example, we will ask it respond as if a weatherman, so to return the temperature and then any other additional info.
#
# This examples builds off the base chat executor. It is highly recommended you learn about that executor before going through this notebook. You can find documentation for that example [here](./base.ipynb).
#
# Any modifications of that example are called below with **MODIFICATION**, so if you are looking for the differences you can just search for that.

# %% [markdown]
# ## Set up the State
#
# The main type of graph in `langgraph` is the [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph).
# This graph is parameterized by a `State` object that it passes around to each node.
# Each node then returns operations the graph uses to `update` that state.
# These operations can either SET specific attributes on the state (e.g. overwrite the existing values) or ADD to the existing attribute.
# Whether to set or add is denoted by annotating the `State` object you use to construct the graph.
#
# For this example, the state we will track will just be a list of messages.
# We want each node to just add messages to that list.
# Therefore, we will use a `TypedDict` with one key (`messages`) and annotate it so that the `messages` attribute is "append-only".

# %%
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

# Add messages essentially does this with more
# robust handling
# def add_messages(left: list, right: list):
#     return left + right


class State(TypedDict):
    messages: Annotated[list, add_messages]


# %% [markdown]
# ## Set up the tools
#
# We will first define the tools we want to use.
# For this simple example, we will use create a placeholder search engine.
# It is really easy to create your own tools - see documentation [here](https://python.langchain.com/v0.2/docs/how_to/custom_tools) on how to do that.
#

# %%
from langchain_core.tools import tool


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    return ["The answer to your question lies within."]


tools = [search]

# %% [markdown]
# We can now wrap these tools in a simple [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode).
# This is  a simple class that takes in a list of messages containing an [AIMessages with tool_calls](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.tool_calls), runs the tools, and returns the output as [ToolMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.ToolMessage.html#langchain_core.messages.tool.ToolMessage)s.
#

# %%
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

# %% [markdown]
# ## Set up the model
#
# Now we need to load the chat model we want to use.
# This should satisfy two criteria:
#
# 1. It should work with messages, since our state is primarily a list of messages (chat history).
# 2. It should work with tool calling, since we are using a prebuilt [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)
#
# **Note:** these model requirements are not requirements for using LangGraph - they are just requirements for this particular example.
#

# %%
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-haiku-20240307")

# %% [markdown]
#
# After we've done this, we should make sure the model knows that it has these tools available to call.
# We can do this by converting the LangChain tools into the format for function calling, and then bind them to the model class.
#

# %%
model = model.bind_tools(tools)

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

# %%
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]

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

# We will set streaming=True so that we can stream tokens
# See the streaming section for more information on this.
model = ChatOpenAI(temperature=0, streaming=True)

# %% [markdown]
#
# After we've done this, we should make sure the model knows that it has these tools available to call.
# We can do this by converting the LangChain tools into the format for OpenAI function calling, and then bind them to the model class.
#
#
# **MODIFICATION**
#
# We also want to define a response schema for the language model and bind it to the model as a function as well

# %%
from langchain_core.pydantic_v1 import BaseModel, Field


class Response(BaseModel):
    """Final response to the user"""

    temperature: float = Field(description="the temperature")
    other_notes: str = Field(description="any other notes about the weather")


model = model.bind_tools(tools + [Response])

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
# **MODIFICATION**
#
# We will change the `should_continue` function to check what function was called. If the function `Response` was called - that is the function that is NOT a tool, but rather the formatted response, so we should NOT continue in that case.

# %%
from typing import Literal

from langchain_core.messages import ToolMessage

from langgraph.prebuilt import ToolInvocation


# Define the function that determines whether to continue or not
def should_continue(state) -> Literal["continue", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we need to check what type of function call it is
    if last_message.tool_calls[0]["name"] == "Response":
        return "end"
    # Otherwise we continue
    return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation for each tool call
    tool_invocations = []
    for tool_call in last_message.tool_calls:
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        tool_invocations.append(action)

    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    # We call the tool_executor and get back a response
    responses = tool_executor.batch(tool_invocations, return_exceptions=True)
    # We use the response to create tool messages
    tool_messages = [
        ToolMessage(
            content=str(response),
            name=tc["name"],
            tool_call_id=tc["id"],
        )
        for tc, response in zip(last_message.tool_calls, responses)
    ]

    # We return a list, because this will get added to the existing list
    return {"messages": tool_messages}


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
workflow.add_node("action", call_tool)

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

# %%
from IPython.display import Image, display

try:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %% [markdown]
# ## Use it!
#
# We can now use it!
# This now exposes the [same interface](https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel) as all other LangChain runnables.

# %%
from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value["messages"][-1])
    print("\n---\n")

# %%

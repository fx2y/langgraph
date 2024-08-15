# %% [markdown]
# # How to manage agent steps
#
# In this example we will build a ReAct Agent that explicitly manages intermediate steps.
#
# The previous examples just put all messages into the model, but that extra context can distract the agent and add latency to the API calls. In this example we will only include the `N` most recent messages in the chat history. Note that this is meant to be illustrative of general state management.

# %% [markdown]
# ## Setup
#
# First we need to install the packages required

# %%
# %%capture --no-stderr
# %pip install --quiet -U langgraph langchain_openai

# %% [markdown]
# Next, we need to set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)

# %%
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")

# %% [markdown]
# Optionally, we can set API key for [LangSmith tracing](https://smith.langchain.com/), which will give us best-in-class observability.

# %%
os.environ["LANGCHAIN_TRACING_V2"] = "true"
_set_env("LANGCHAIN_API_KEY")

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
    return [
        "Try again in a few seconds! Checking with the weathermen... Call be again next."
    ]


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
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# %% [markdown]
#
# After we've done this, we should make sure the model knows that it has these tools available to call.
# We can do this by converting the LangChain tools into the format for function calling, and then bind them to the model class.
#

# %%
model = model.bind_tools(tools)

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

# %%
from typing import Literal


# Define the function that determines whether to continue or not
def should_continue(state: State) -> Literal["__end__", "action"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# %% [markdown]
# **MODIFICATION**
#
# Here we don't pass all messages to the model but rather only pass the `N` most recent. Note that this is a terribly simplistic way to handle messages meant as an illustrtion, and there may be other methods you may want to look into depending on your use case. We also have to make sure we don't truncate the chat history to include the tool message first, as this would cause an API error.

# %%
# Define the function that calls the model
def call_model(state):
    messages = []
    for m in state["messages"][::-1]:
        messages.append(m)
        if len(messages) >= 5:
            if messages[-1].type != "tool":
                break
    response = model.invoke(messages[::-1])
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# %% [markdown]
# ## Define the graph
#
# We can now put it all together and define the graph!

# %%
from langgraph.graph import END, StateGraph, START

# Define a new graph
workflow = StateGraph(State)

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

# %%
from IPython.display import Image, display

display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# %% [markdown]
# ## Use it!
#
# We can now use it!
# This now exposes the [same interface](https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel) as all other LangChain runnables.

# %%
from langchain_core.messages import HumanMessage

inputs = {
    "messages": [
        HumanMessage(
            content="what is the weather in sf? Don't give up! Keep using your tools."
        )
    ]
}
for event in app.stream(inputs, stream_mode="values"):
    # stream() yields dictionaries with output keyed by node name
    for message in event["messages"]:
        message.pretty_print()
    print("\n---\n")

# %%

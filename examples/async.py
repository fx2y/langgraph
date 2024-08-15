# %% [markdown]
# # How to run graph asynchronously
#
# In this example we will build a ReAct agent with native [async](https://docs.python.org/3/library/asyncio.html) implementations of the core logic. When chat models have async clients, this can give us some nice performance improvements if you
# are running concurrent branches in your graph or if your graph is running within a larger web server process.
#
# In general, you don't need to change anything about your graph to add `async` support. That's one of the beauties of [Runnables](https://python.langchain.com/docs/expression_language/interface/). 
#
#
# <div class="admonition tip">
#     <p class="admonition-title">Note:</p>
#     <p>
#         In this how-to, we will create our agent from scratch to be transparent (but verbose). You can accomplish similar functionality using the <code>create_react_agent(model, tools=tool)</code> (<a href="https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent">API doc</a>) constructor. This may be more appropriate if you are used to LangChain’s <a href="https://python.langchain.com/v0.1/docs/modules/agents/concepts/#agentexecutor">AgentExecutor</a> class.
#     </p>
# </div>    

# %% [markdown]
# ## Setup
#
# First we need to install the packages required

# %%
# %%capture --no-stderr
# %pip install --quiet -U langgraph langchain_anthropic

# %% [markdown]
# Next, we need to set API keys for Anthropic (the LLM we will use).

# %%
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")

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
# It is really easy to create your own tools - see documentation [here](https://python.langchain.com/docs/modules/agents/tools/custom_tools) on how to do that.
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
# **MODIFICATION**
#
# We define each node as an async function.

# %%
from typing import Literal


# Define the function that determines whether to continue or not
def should_continue(state: State) -> Literal["end", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
async def call_model(state: State):
    messages = state["messages"]
    response = await model.ainvoke(messages)
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

display(Image(app.get_graph().draw_mermaid_png()))

# %% [markdown]
# ## Use it!
#
# We can now use it!
# This now exposes the [same interface](https://python.langchain.com/docs/expression_language/) as all other LangChain runnables.

# %%
from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
await app.ainvoke(inputs)

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
async for output in app.astream(inputs, stream_mode="updates"):
    # stream_mode="updates" yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value["messages"][-1].pretty_print())
    print("\n---\n")

# %% [markdown]
# ### Streaming LLM Tokens
#
# You can also access the LLM tokens as they are produced by each node. 
# In this case only the "agent" node produces LLM tokens.
# In order for this to work properly, you must be using an LLM that supports streaming as well as have set it when constructing the LLM (e.g. `ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)`)
#

# %%
inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
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
            print(op["value"].content, end="|")

# %%

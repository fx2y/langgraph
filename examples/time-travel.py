# %% [markdown]
# # How to view and update graph state
#
# Once you start [checkpointing](./persistence.ipynb) your graphs, you can easily **get** or **update** the state of the agent at any point in time. This permits a few things:
#
# 1. You can surface a state during an interrupt to a user to let them accept an action.
# 2. You can **rewind** the graph to reproduce or avoid issues.
# 3. You can **modify** the state to embed your agent into a larger system, or to let the user better control its actions.
#
# The key methods used for this functionality are:
#
# - [get_state](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.get_state): fetch the values from the target config
# - [update_state](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.update_state): apply the given values to the target state
#
# **Note:** this requires passing in a checkpointer.
#
# Below is a quick example.
#
# <div class="admonition tip">
#     <p class="admonition-title">Note:</p>
#     <p>
#         In this how-to, we will create our agent from scratch to be transparent (but verbose). You can accomplish similar functionality using the <code>create_react_agent(model, tools=tool, checkpointer=checkpointer)</code> (<a href="https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent">API doc</a>) constructor. This may be more appropriate if you are used to LangChain’s <a href="https://python.langchain.com/v0.2/docs/how_to/agent_executor/#concepts">AgentExecutor</a> class.
#     </p>
# </div>    

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


_set_env("ANTHROPIC_API_KEY")

# %% [markdown]
# Optionally, we can set API key for [LangSmith tracing](https://smith.langchain.com/), which will give us best-in-class observability.

# %%
os.environ["LANGCHAIN_TRACING_V2"] = "true"
_set_env("LANGCHAIN_API_KEY")

# %% [markdown]
# ## Set up the State
#
# The state is the interface for all the nodes.

# %%
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

# `add_messages`` essentially does this
# (with more robust handling)
# def add_messages(left: list, right: list):
#     return left + right


class State(TypedDict):
    messages: Annotated[list, add_messages]


# %% [markdown]
# ## Set up the tools
#
# We will first define the tools we want to use.
# For this simple example, we will use create a placeholder search engine.
# However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/v0.2/docs/how_to/custom_tools) on how to do that.
#

# %%
from langchain_core.tools import tool


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    return ["The weather is cloudy with a chance of meatballs."]


tools = [search]

# %% [markdown]
# We can now wrap these tools in a simple ToolNode.
# This is a prebuilt node that extracts tool calls from the most recent AIMessage, executes them, and returns a ToolMessage with the results.
#

# %%
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

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

# %%
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)

# %% [markdown]
#
# After we've done this, we should make sure the model knows that it has these tools available to call.
# We can do this using the `.bind_tools()` method, common to many of LangChain's chat models.
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

# %%
from typing import Literal


# Define the function that determines whether to continue or not
def should_continue(state: State) -> Literal["continue", "end"]:
    last_message = state["messages"][-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# %% [markdown]
# ## Define the graph
#
# We can now put it all together and define the graph!

# %%
from langgraph.graph import END, StateGraph, START

# Define a new graph
workflow = StateGraph(State)


# Define the two nodes we will cycle between
def call_model(state: State) -> State:
    return {"messages": model.invoke(state["messages"])}


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

# %% [markdown]
# **Persistence**
#
# To add in persistence, we pass in a checkpoint when compiling the graph

# %%
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# %%
# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile(checkpointer=memory)

# %% [markdown]
# ## Preview the graph

# %%
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %% [markdown]
# ## Interacting with the Agent
#
# We can now interact with the agent. Between interactions you can get and update state.
#

# %%
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %% [markdown]
# See LangSmith example run here https://smith.langchain.com/public/01c1d61c-6943-4db1-8afe-5366f083caf3/r
#
# Here you can see the "agent" node ran, and then "should_continue" returned "end" so the graph stopped execution there.

# %% [markdown]
# Let's now get the current state

# %%
app.get_state(config).values

# %% [markdown]
# The current state is the two messages we've seen above, 1. the HumanMessage we sent in, 2. the AIMessage we got back from the model.
#
# The `next` values are empty since the graph has terminated (transitioned to the `__end__`).

# %%
app.get_state(config).next

# %% [markdown]
# The graph got to the end without interruptions, so the list of next nodes is empty.

# %% [markdown]
# ### Let's get it to execute a tool

# %%
config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="what is the weather in sf currently")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %% [markdown]
# See LangSmith example run here https://smith.langchain.com/public/c33c04c5-f1f2-4977-9d7d-c48f28be7be2/r

# %% [markdown]
# We can see it planned the tool execution (ie the "agent" node), then "should_continue" edge returned "continue" so we proceeded to "action" node, which executed the tool, and then "agent" node emitted the final response, which made "should_continue" edge return "end". Let's see how we can have more control over this.

# %% [markdown]
# ### Pause before tools

# %% [markdown]
# If you notice below, we now will add `interrupt_before=["action"]` - this means that before any actions are taken we pause. This is a great moment to allow the user to correct and update the state! This is very useful when you want to have a human-in-the-loop to validate (and potentially change) the action to take. 

# %%
app_w_interrupt = workflow.compile(checkpointer=memory, interrupt_before=["action"])

# %%
config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="what is the weather in sf currently")
for event in app_w_interrupt.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    event["messages"][-1].pretty_print()

# %% [markdown]
# See LangSmith example run here https://smith.langchain.com/public/22402055-a50e-4d82-8b3e-733c9d752bc5/r
# This time it executed the "agent" node same as before, and you can see in the LangSmith trace that "should_continue" returned "continue", but it paused execution per our setting above.

# %% [markdown]
# Notice that this time, the `next` value is populated with `action`. That means that if we resume the graph, it will start at the `action` node.

# %%
current_values = app_w_interrupt.get_state(config)
current_values.next

# %% [markdown]
# Because we asked to interrupt the graph before getting to the action node, the next node to execute, if we were to resume, would be the "action" node.

# %%
current_values.values["messages"][-1].tool_calls

# %% [markdown]
# Let's update the search string before proceeding

# %%
current_values.values["messages"][-1].tool_calls[0]["args"][
    "query"
] = "weather in San Francisco today"

# %%
app_w_interrupt.update_state(config, current_values.values)

# %% [markdown]
# This actually produces a LangSmith run too! See it here https://smith.langchain.com/public/9d86718b-333e-4175-bec0-9a64cdd01dc3/r
#
# This is a shorter run that allows you to inspect the edges that reacted to the state update, you can see "should_continue" returned "continue" as before, given this is still a function call.

# %% [markdown]
# The current state now reflects our updated search query!

# %%
app_w_interrupt.get_state(config).values

# %%
app_w_interrupt.get_state(config).next

# %% [markdown]
# If we start the agent again it will pick up from the state we updated.

# %%
for event in app_w_interrupt.stream(None, config):
    for v in event.values():
        print(v)

# %% [markdown]
# See this run in LangSmith here https://smith.langchain.com/public/8262c0f9-0701-4d73-95f6-2a32f6d3f96a/r
#
# This continues where we left off, with "action" node, followed by "agent" node, which terminates the execution.

# %% [markdown]
# ## Checking history
#
# Let's browse the history of this thread, from newest to oldest.

# %%
for state in app_w_interrupt.get_state_history(config):
    print(state)
    print("--")
    if len(state.values["messages"]) == 2:
        to_replay = state

# %% [markdown]
# We can go back to any of these states and restart the agent from there!

# %%
to_replay.values

# %%
to_replay.next

# %% [markdown]
# ### Replay a past state
#
# To replay from this place we just need to pass its config back to the agent.

# %%
for event in app_w_interrupt.stream(None, to_replay.config):
    for v in event.values():
        print(v)

# %% [markdown]
# See this run in LangSmith here https://smith.langchain.com/public/f26e9e1d-16df-48ae-98f7-c823d6942bf7/r
#
# This is similar to the previous run, this time with the original search query, instead of our modified one. 

# %% [markdown]
# ### Branch off a past state
#
# Using LangGraph's checkpointing, you can do more than just replay past states. You can branch off previous locations to let the agent explore alternate trajectories or to let a user "version control" changes in a workflow.

# %%
from langchain_core.messages import AIMessage

branch_config = app_w_interrupt.update_state(
    to_replay.config,
    {
        "messages": [
            AIMessage(content="All done here!", id=to_replay.values["messages"][-1].id)
        ]
    },
)

# %%
branch_state = app_w_interrupt.get_state(branch_config)

# %%
branch_state.values

# %%
branch_state.next

# %% [markdown]
# You can see the snapshot was updated and now correctly reflects that there is no next step.
#
# You can see this in LangSmith update run here https://smith.langchain.com/public/65104717-6eda-4a0f-93c1-4755c6f929ed/r
#
# This shows the "should_continue" edge now reacting to this replaced message, and now changing the outcome to "end" which finishes the computation.

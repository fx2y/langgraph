# %% [markdown]
# # How to edit graph state
#
# Human-in-the-loop (HIL) interactions are crucial for [agentic systems](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#human-in-the-loop). Manually updating the graph state a common HIL interaction pattern, allowing the human to edit actions (e.g., what tool is being called or how it is being called).
#
# We can implement this in LangGraph using a [breakpoint](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/): breakpoints allow us to interrupt graph execution before a specific step. At this breakpoint, we can manually update the graph state and then resume from that spot to continue.  
#
# ![image.png](attachment:49539520-097a-43d5-94b4-2b56193a579f.png)

# %% [markdown]
# ## Setup
#
# First we need to install the packages required

# %%
# %%capture --no-stderr
# %pip install --quiet -U langgraph langchain_anthropic

# %% [markdown]
# Next, we need to set API keys for Anthropic (the LLM we will use)

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
# ## Simple Usage
#
# Let's look at very basic usage of this.
#
# Below, we do three things:
#
# 1) We specify the [breakpoint](https://langchain-ai.github.io/langgraph/concepts/low_level/#breakpoints) using `interrupt_before` a specified step (node).
#
# 2) We set up a [checkpointer](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer) to save the state of the graph up until this node.
#
# 3) We use `.update_state` to update the state of the graph.

# %%
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display


class State(TypedDict):
    input: str


def step_1(state):
    print("---Step 1---")
    pass


def step_2(state):
    print("---Step 2---")
    pass


def step_3(state):
    print("---Step 3---")
    pass


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Add
graph = builder.compile(checkpointer=memory, interrupt_before=["step_2"])

# View
display(Image(graph.get_graph().draw_mermaid_png()))

# %%
# Input
initial_input = {"input": "hello world"}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)

# %% [markdown]
# Now, we can just manually update our graph state - 

# %%
print("Current state!")
print(graph.get_state(thread).values)

graph.update_state(thread, {"input": "hello universe!"})

print("---\n---\nUpdated state!")
print(graph.get_state(thread).values)

# %%
# Continue the graph execution
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)

# %% [markdown]
# ## Agent
#
# In the context of agents, updating state is useful for things like editing tool calls.
#  
# To show this, we will build a relatively simple ReAct-style agent that does tool calling. 
#
# We will use Anthropic's models and a fake tool (just for demo purposes).

# %%
# Set up the tool
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return [
        "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    ]


tools = [search]
tool_node = ToolNode(tools)

# Set up the model

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
model = model.bind_tools(tools)


# Define nodes and conditional edges


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
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


# Define a new graph
workflow = StateGraph(MessagesState)

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

# Set up memory
memory = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable

# We add in `interrupt_before=["action"]`
# This will add a breakpoint before the `action` node is called
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])

# %% [markdown]
# ## Interacting with the Agent
#
# We can now interact with the agent and see that it stops before calling a tool.
#

# %%
from langchain_core.messages import HumanMessage

thread = {"configurable": {"thread_id": "3"}}
inputs = [HumanMessage(content="search for the weather in sf now")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %% [markdown]
# **Edit**
#
# We can now update the state accordingly. Let's modify the tool call to have the query `"current weather in SF"`.

# %%
# First, lets get the current state
current_state = app.get_state(thread)

# Let's now get the last message in the state
# This is the one with the tool calls that we want to update
last_message = current_state.values["messages"][-1]

# Let's now update the args for that tool call
last_message.tool_calls[0]["args"] = {"query": "current weather in SF"}

# Let's now call `update_state` to pass in this message in the `messages` key
# This will get treated as any other update to the state
# It will get passed to the reducer function for the `messages` key
# That reducer function will use the ID of the message to update it
# It's important that it has the right ID! Otherwise it would get appended
# as a new message
app.update_state(thread, {"messages": last_message})

# %% [markdown]
# Let's now check the current state of the app to make sure it got updated accordingly

# %%
current_state = app.get_state(thread).values["messages"][-1].tool_calls
current_state

# %% [markdown]
# **Resume**
#
# We can now call the agent again with no inputs to continue, ie. run the tool as requested. We can see from the logs that it passes in the update args to the tool.

# %%
for event in app.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %%

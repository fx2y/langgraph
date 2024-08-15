# %% [markdown]
# # How to add human-in-the-loop
#
# When creating LangGraph agents, it is often nice to add a human-in-the-loop component.
# This can be helpful when giving them access to tools.
# Often in these situations you may want to manually approve an action before taking.
#
# This can be in several ways, but the primary supported way is to add an "interrupt" before a node is executed.
# This interrupts execution at that node.
# You can then resume from that spot to continue.
#
# <div class="admonition tip">
#     <p class="admonition-title">Note</p>
#     <p>
#         In this how-to, we will create our agent from scratch to be transparent (but verbose). You can accomplish similar functionality using either `interrupt_before` or `interrupt_after` in the <code>create_react_agent(model, tools=tool, interrupt_before=["tools" | "agent"], interrupt_after=["tools" | "agent"])</code> (<a href="https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent">API doc</a>) constructor. This may be more appropriate if you are used to LangChain’s <a href="https://python.langchain.com/v0.2/docs/how_to/agent_executor/#concepts">AgentExecutor</a> class.
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


_set_env("OPENAI_API_KEY")

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
# However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/docs/modules/agents/tools/custom_tools) on how to do that.
#

# %%
from langchain_core.tools import tool


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return [
        "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    ]


tools = [search]

# %% [markdown]
# We can now wrap these tools in a simple [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode).
# This is  a simple class that takes in a list of messages containing an [AIMessages with tool_calls](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.tool_calls), runs the tools, and returns the output as [ToolMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.ToolMessage.html#langchain_core.messages.tool.ToolMessage)s.
# A ToolInvocation is any class with `tool` and `tool_input` attribute.
#

# %%
from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(tools)

# %% [markdown]
# ## Set up the model
#
# Now we need to load the chat model we want to use.
# Since we are creating a tool-using ReAct agent, we want to make sure the model supports [Tool Calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/) and works with chat messages.
#
# Note: these model requirements are not requirements for using LangGraph - they are just requirements for this one example.

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

# %%
from langchain_core.messages import ToolMessage

from langgraph.prebuilt import ToolInvocation


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


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
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

# %%
from langgraph.graph import END, StateGraph, START

# Define a new graph
workflow = StateGraph(State)

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

# %% [markdown]
# **Persistence**
#
# To add in persistence, we pass in a checkpoint when compiling the graph. Persistence is required to support interrupts, since the graph will stop executing while it is interrupted.

# %%
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# %% [markdown]
# **Interrupt**
#
# To always interrupt before a particular node, pass the name of the node to compile.

# %%
# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])

# %% [markdown]
# ## Preview the graph

# %%
from IPython.display import Image, display

display(Image(app.get_graph().draw_mermaid_png()))

# %% [markdown]
# ## Interacting with the Agent
#
# We can now interact with the agent and see that it stops before calling a tool.
#

# %%
from langchain_core.messages import HumanMessage

thread = {"configurable": {"thread_id": "2"}}
inputs = [HumanMessage(content="hi! I'm bob")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %%
inputs = [HumanMessage(content="What did I tell you my name was?")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %%
inputs = [HumanMessage(content="what's the weather in sf now?")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %% [markdown]
# **Resume**
#
# We can now call the agent again with no inputs to continue, ie. run the tool as requested.
#
# Running an interrupted graph with `None` in the inputs means to "proceed as if the interruption didn't occur."

# %%
for event in app.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %% [markdown]
# ## Conversational human-in-the-loop
#
# Suppose that upon interruption, we wish to intervene in the agent's action. How should we implement an intervention?
#
# There are multiple options, and the ideal option may depend on the specifics of your application and capabilities of your chosen LLM. Note that many chat models require that messages with tool calls be immediately followed by a tool message containing the result of the tool call. So our intervention may:
#
# 1. Update the parameters of the tool call before proceeding normally (see this [how-to guide](./human_in_the_loop/breakpoints.ipynb) for an example);
# 2. Add a tool message to the conversation history indicating the user's desired intervention (see an example [here](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/#state-assistant));
# 3. Catch the tool call message, replacing it with a `AIMessage` asking for verification and only adding the tool call message to the conversation history if approved.
#
# Below we demonstrate the third option, supporting a conversational human-in-the-loop experience in which the user can instruct the LLM to modify tool calls before execution via a typical chat interface. We include two implementations-- one in which we interrupt and manually update the state, and one in which we customize the state of the underlying graph.
#
# ### Manually update state
#
# Once the graph execution is interrupted, we are free to issue arbitrary updates to the state. Below, if a tool call is generated, we will:
#
# 1. Append a "verification" `AIMessage` to the state asking for user approval;
# 2. Receive user input and append it to the state as a `HumanMessage`;
# 3. If approved, append the tool call message to the state and resume execution;
# 4. Otherwise, resume execution from the new user input.

# %%
import json
from typing import Optional

from langchain_core.messages import AIMessage


# Helper function to construct message asking for verification
def generate_verification_message(message: AIMessage) -> None:
    """Generate "verification message" from message with tool calls."""
    serialized_tool_calls = json.dumps(
        message.tool_calls,
        indent=2,
    )
    return AIMessage(
        content=(
            "I plan to invoke the following tools, do you approve?\n\n"
            "Type 'y' if you do, anything else to stop.\n\n"
            f"{serialized_tool_calls}"
        ),
        id=message.id,
    )


# Helper function to stream output from the graph
def stream_app_catch_tool_calls(inputs, thread) -> Optional[AIMessage]:
    """Stream app, catching tool calls."""
    tool_call_message = None
    for event in app.stream(inputs, thread, stream_mode="values"):
        message = event["messages"][-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_call_message = message
        else:
            message.pretty_print()

    return tool_call_message


# %%
import uuid

thread = {"configurable": {"thread_id": "3"}}

tool_call_message = stream_app_catch_tool_calls(
    {"messages": [HumanMessage("what's the weather in sf now?")]},
    thread,
)

while tool_call_message:
    verification_message = generate_verification_message(tool_call_message)
    verification_message.pretty_print()
    input_message = HumanMessage(input())
    if input_message.content == "exit":
        break
    input_message.pretty_print()

    # First we update the state with the verification message and the input message.
    # note that `generate_verification_message` sets the message ID to be the same
    # as the ID from the original tool call message. Updating the state with this
    # message will overwrite the previous tool call.
    snapshot = app.get_state(thread)
    snapshot.values["messages"] += [verification_message, input_message]

    if input_message.content == "y":
        tool_call_message.id = str(uuid.uuid4())
        # If verified, we append the tool call message to the state
        # and resume execution.
        snapshot.values["messages"] += [tool_call_message]
        app.update_state(thread, snapshot.values, as_node="agent")
    else:
        # Otherwise, resume execution from the input message.
        app.update_state(thread, snapshot.values, as_node="__start__")

    tool_call_message = stream_app_catch_tool_calls(None, thread)


# %% [markdown]
# ### Customize the state
#
# Alternatively, we can handle the verification inside the graph, without interrupting execution. We only need to make two changes to the original graph:
#
# 1. We add a key to the state where we will cache tool calls generated by the LLM;
# 2. When calling the LLM, if a tool call message is generated we will cache it and generate a verification message instead. If the tool call is verified, we will return the cached message.

# %%
class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_message: Optional[AIMessage]


def call_model(state):
    messages = state["messages"]
    if messages[-1].content == "y":
        return {
            "messages": [state["tool_call_message"]],
            "tool_call_message": None,
        }
    else:
        response = model.invoke(messages)
        if response.tool_calls:
            verification_message = generate_verification_message(response)
            response.id = str(uuid.uuid4())
            return {
                "messages": [verification_message],
                "tool_call_message": response,
            }
        else:
            return {
                "messages": [response],
                "tool_call_message": None,
            }


# %% [markdown]
# We then build and compile the graph exactly as before:

# %%
workflow = StateGraph(State)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

app = workflow.compile(checkpointer=memory)

# %%
thread = {"configurable": {"thread_id": "4"}}

inputs = [HumanMessage(content="what's the weather in sf?")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %%
inputs = [HumanMessage(content="can you specify sf in CA?")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %%
inputs = [HumanMessage(content="y")]
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

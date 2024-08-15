# %% [markdown]
# # How to stream events from within a tool

# %% [markdown]
# If your LangGraph graph needs to use tools that call LLMs (or any other LangChain `Runnable` objects -- other graphs, LCEL chains, retrievers, etc.), you might want to stream events from the underlying `Runnable`. This guide shows how you can do that.

# %% [markdown]
# ## Setup

# %%
# %%capture --no-stderr
# %pip install -U langgraph langchain-openai

# %%
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")

# %% [markdown]
# ## Define graph and tools

# %% [markdown]
# We'll use a prebuilt ReAct agent for this guide

# %%
from langchain_core.callbacks import Callbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# %% [markdown]
# <div class="admonition warning">
#     <p class="admonition-title">ASYNC IN PYTHON<=3.10</p>
#     <p>
# Any Langchain RunnableLambda, a RunnableGenerator, or Tool that invokes other runnables and is running async in python<=3.10, will have to propagate callbacks to child objects manually. This is because LangChain cannot automatically propagate callbacks to child objects in this case.
#     
# This is a common reason why you may fail to see events being emitted from custom runnables or tools.
#     </p>
# </div>

# %%
@tool
async def get_items(
    place: str, callbacks: Callbacks
) -> str:  # <--- Accept callbacks (Python <= 3.10)
    """Use this tool to look up which items are in the given place."""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "Can you tell me what kind of items i might find in the following place: '{place}'. "
                "List at least 3 such items separating them by a comma. And include a brief description of each item..",
            )
        ]
    )
    chain = template | llm.with_config(
        {
            "run_name": "Get Items LLM",
            "tags": ["tool_llm"],
            "callbacks": callbacks,  # <-- Propagate callbacks (Python <= 3.10)
        }
    )
    chunks = [chunk async for chunk in chain.astream({"place": place})]
    return "".join(chunk.content for chunk in chunks)

# %% [markdown]
# We're adding a custom tag (`tool_llm`) to our LLM runnable within the tool. This will allow us to filter events that we'll stream from the compiled graph (`agent`) Runnable below

# %%
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
tools = [get_items]
agent = create_react_agent(llm, tools=tools)

# %% [markdown]
# ## Stream events from the graph

# %%
async for event in agent.astream_events(
    {"messages": [("human", "what items are on the shelf?")]}, version="v2"
):
    tags = event.get("tags", [])
    if event["event"] == "on_chat_model_stream" and "tool_llm" in tags:
        print(event["data"]["chunk"].content, end="", flush=True)

# %% [markdown]
# Let's inspect the last event to get the final list of messages from the agent

# %%
final_messages = event["data"]["output"]["messages"]

# %%
for message in final_messages:
    message.pretty_print()

# %% [markdown]
# You can see that the content of the `ToolMessage` is the same as the output we streamed above

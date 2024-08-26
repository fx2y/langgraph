# %% [markdown]
# # How to stream arbitrary nested content
#
# The most common use case for streaming from inside a node is to stream LLM tokens, but you may have other long-running streaming functions you wish to render for the user. While individual nodes in LangGraph cannot return generators (since they are executed to completion for each [superstep](https://langchain-ai.github.io/langgraph/concepts/#core-design)), we can still stream arbitrary custom functions from within a node using a similar tact and calling `astream_events` on the graph.
#
# We do so using a [RunnableGenerator](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.RunnableGenerator.html#langchain-core-runnables-base-runnablegenerator) (which your function will automatically behave as if wrapped as a [RunnableLambda](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.RunnableLambda.html#langchain_core.runnables.base.RunnableLambda)).
#
# Below is a simple toy example.

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
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableGenerator
from langchain_core.runnables import RunnableConfig

from langgraph.graph import START, StateGraph, MessagesState, END

# Define a new graph
workflow = StateGraph(MessagesState)


async def my_generator(state: MessagesState):
    messages = [
        "Four",
        "score",
        "and",
        "seven",
        "years",
        "ago",
        "our",
        "fathers",
        "...",
    ]
    for message in messages:
        yield message


async def my_node(state: MessagesState, config: RunnableConfig):
    messages = []
    # Tagging a node makes it easy to filter out which events to include in your stream
    # It's completely optional, but useful if you have many functions with similar names
    gen = RunnableGenerator(my_generator).with_config(
        tags=["should_stream"],
        callbacks=config.get(
            "callbacks", []
        ),  # <-- Propagate callbacks (Python <= 3.10)
    )
    async for message in gen.astream(state):
        messages.append(message)
    return {"messages": [AIMessage(content=" ".join(messages))]}


workflow.add_node("model", my_node)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)
app = workflow.compile()

# %%
from langchain_core.messages import HumanMessage

inputs = [HumanMessage(content="What are you thinking about?")]
async for event in app.astream_events({"messages": inputs}, version="v2"):
    kind = event["event"]
    tags = event.get("tags", [])
    if kind == "on_chain_stream" and "should_stream" in tags:
        data = event["data"]
        if data:
            # Empty content in the context of OpenAI or Anthropic usually means
            # that the model is asking for a tool to be invoked.
            # So we only print non-empty content
            print(data, end="|")

# %%

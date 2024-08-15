# %% [markdown]
# # Quickstart
#

# %%
# %%capture --no-stderr
# %pip install --quiet -U langgraph langchain-openai

# %%
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# %%
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import END, MessageGraph

model = ChatOpenAI(temperature=0)

graph = MessageGraph()

graph.add_node("oracle", model)
graph.add_edge("oracle", END)

graph.add_edge(START, "oracle")

runnable = graph.compile()

# %%
from IPython.display import Image, display

try:
    display(Image(runnable.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
runnable.invoke(HumanMessage("What is 1 + 1?"))

# %%
from typing import Literal

from langchain_core.tools import tool

from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode


@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number


model = ChatOpenAI(temperature=0)
model_with_tools = model.bind_tools(tools=[multiply])

graph = MessageGraph()

graph.add_node("oracle", model_with_tools)

tool_node = ToolNode([multiply])
graph.add_node("multiply", tool_node)
graph.add_edge(START, "oracle")
graph.add_edge("multiply", END)


def router(state: list[BaseMessage]) -> Literal["multiply", "__end__"]:
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    if len(tool_calls):
        return "multiply"
    else:
        return END


graph.add_conditional_edges("oracle", router)
runnable = graph.compile()

# %%
try:
    display(Image(runnable.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
runnable.invoke(HumanMessage("What is 123 * 456?"))

# %%
runnable.invoke(HumanMessage("What is your name?"))

# %%

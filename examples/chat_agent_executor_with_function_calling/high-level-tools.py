# %% [markdown]
# # ReAct agent with tool calling
#
# This notebook walks through an example creating a ReAct Agent that uses tool calling.
# This is useful for getting started quickly.
# However, it is highly likely you will want to customize the logic - for information on that, check out the other examples in this folder.

# %% [markdown]
# ## Set up the chat model and tools
#
# Here we will define the chat model and tools that we want to use.
# Importantly, this model MUST support OpenAI function calling.

# %%
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent

# %%
tools = [TavilySearchResults(max_results=1)]
model = ChatOpenAI()

# %% [markdown]
# ## Create executor
#
# We can now use the high level interface to create the executor

# %%
app = create_react_agent(model, tools=tools)

# %% [markdown]
# We can now invoke this executor. The input to this must be a dictionary with a single `messages` key that contains a list of messages.

# %%
inputs = {"messages": [HumanMessage(content="what is the weather in sf and la")]}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")

# %%

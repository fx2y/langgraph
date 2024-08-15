# %% [markdown]
# # How to create agents with configuration
#
# One of the benefits of LangGraph API is that it lets you create agents with different configurations.
# This is useful when you want to:
#
# - Define a cognitive architecture once as a LangGraph
# - Let that LangGraph be configurable across some attributes (for example, system message or LLM to use)
# - Let users create agents with arbitrary configurations, save them, and then use them in the future
#
# In this guide we will show how to do that for the default agent we have built in.
#
# If you look at the agent we defined, you can see that inside the `call_model` node we have created the model based on some configuration. That node looks like:
#
# ```python
# def call_model(state, config):
#     messages = state["messages"]
#     model_name = config.get('configurable', {}).get("model_name", "anthropic")
#     model = _get_model(model_name)
#     response = model.invoke(messages)
#     # We return a list, because this will get added to the existing list
#     return {"messages": [response]}
# ```
#
# We are looking inside the config for a `model_name` parameter (which defaults to `anthropic` if none is found).
# That means that by default we are using Anthropic as our model provider.
# In this example we will see an example of how to create an example agent that is configured to use OpenAI.
#
# We've also communicated to the graph that it should expect configuration with this key. 
# We've done this by passing `config_schema` when constructing the graph, eg:
#
# ```python
# class GraphConfig(TypedDict):
#     model_name: Literal["anthropic", "openai"]
#
#
# # Define a new graph
# workflow = StateGraph(AgentState, config_schema=GraphConfig)
# ```

# %%
from langgraph_sdk import get_client

client = get_client()

# %%
# First, let's check what valid configuration can be
# We can do this by getting the default assistant
# There should always be a default assistant with no configuration
assistants = await client.assistants.search()
assistants = [a for a in assistants if not a["config"]]
base_assistant = assistants[0]

# %%
# We can now call `.get_schemas` to get schemas associated with this graph
schemas = await client.assistants.get_schemas(
    assistant_id=base_assistant["assistant_id"]
)
# There are multiple types of schemas
# We can get the `config_schema` to look at the the configurable parameters
schemas["config_schema"]["definitions"]["Configurable"]["properties"]

# %%
assistant = await client.assistants.create(
    graph_id="agent", config={"configurable": {"model_name": "openai"}}
)

# %% [markdown]
# We can see that this assistant has saved the config

# %%
assistant

# %%
thread = await client.threads.create()
input = {"messages": [{"role": "user", "content": "who made you?"}]}
async for event in client.runs.stream(
    thread["thread_id"], assistant["assistant_id"], input=input
):
    print(event)

# %%

# %% [markdown]
# # How to run multiple agents on the same thread
#
# In LangGraph Cloud, a thread is not explicitly associated with a particular agent.
# This means that you can run multiple agents on the same thread, which allows a different
# agent to continue from an initial agent's progress.
#
# In this example, we will create two agents and then call them both on the same thread.
# You'll see that the second agent will respond using information from the [checkpoint](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer-state) generated in the thread
# by the first agent as context.

# %%
from langgraph_sdk import get_client

client = get_client()

openai_assistant = await client.assistants.create(
    graph_id="agent", config={"configurable": {"model_name": "openai"}}
)

# There should always be a default assistant with no configuration
assistants = await client.assistants.search()
default_assistant = [a for a in assistants if not a["config"]][0]

# %% [markdown]
# We can see that these agents are different:

# %%
openai_assistant

# %%
default_assistant

# %% [markdown]
# We can now run the OpenAI assistant on the thread first.

# %%
thread = await client.threads.create()
input = {"messages": [{"role": "user", "content": "who made you?"}]}
async for event in client.runs.stream(
    thread["thread_id"],
    openai_assistant["assistant_id"],
    input=input,
    stream_mode="updates",
):
    print(event)

# %% [markdown]
# Now, we can run it on a second Anthropic-based assistant and see that this second assistant is aware of the initial question, and can answer the question, `and you?`:

# %%
input = {"messages": [{"role": "user", "content": "and you?"}]}
async for event in client.runs.stream(
    thread["thread_id"],
    default_assistant["assistant_id"],
    input=input,
    stream_mode="updates",
):
    print(event)

# %%

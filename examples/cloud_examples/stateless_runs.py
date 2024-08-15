# %% [markdown]
# # Stateless Runs
#
# Most of the time, you provide a `thread_id` to your client when you run your graph in order to keep track of prior runs through the persistent state implemented in LangGraph Cloud. However, if you have your own database to save runs and don't need to use the built in persistent state, you can create stateless runs.
#
# ## Setup
#
# First, let's setup our client

# %%
from langgraph_sdk import get_client

client = get_client()
assistants = await client.assistants.search()
assistants = [a for a in assistants if not a["config"]]
assistant = assistants[0]
thread = await client.threads.create()

# %% [markdown]
# ## Stateless streaming
#
# We can stream the results of a stateless run in an almost identical fashion to how we stream from a run with the state attribute, but instead of passing a value to the `thread_id` parameter, we pass `None`:

# %%
input = {
    "messages": [
        {"role": "user", "content": "Hello! My name is Bagatur and I am 26 years old."}
    ]
}


async for chunk in client.runs.stream(
    # Don't pass in a thread_id and the stream will be stateless
    None,
    assistant["assistant_id"],  # graph_id
    input=input,
    stream_mode="updates",
):
    if chunk.data and "run_id" not in chunk.data:
        print(chunk.data)

# %% [markdown]
# ## Waiting for stateless results
#
# In addition to streaming, you can also wait for a stateless result by using the `.wait` function like follows:

# %%
stateless_run_result = await client.runs.wait(
    None,
    assistant["assistant_id"],  # graph_id
    input=input,
)

# %%
stateless_run_result

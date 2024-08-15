# %% [markdown]
# # How to kick off background runs
#
# This guide covers how to kick off background runs for your agent.
# This can be useful for long running jobs.

# %%
# Initialize the client
from langgraph_sdk import get_client

client = get_client()

# %%
# List available assistants
assistants = await client.assistants.search()
assistants[0]

# %%
# NOTE: we can use `assistant_id` UUID from the above response, or just pass graph ID instead when creating runs. we'll use graph ID here
assistant_id = "agent"

# %%
# Create a new thread
thread = await client.threads.create()
thread

# %%
# If we list runs on this thread, we can see it is empty
runs = await client.runs.list(thread["thread_id"])
runs

# %%
# Let's kick off a run
input = {"messages": [{"role": "human", "content": "what's the weather in sf"}]}
run = await client.runs.create(thread["thread_id"], assistant_id, input=input)

# %%
# The first time we poll it, we can see `status=pending`
await client.runs.get(thread["thread_id"], run["run_id"])

# %%
# Wait until the run finishes
await client.runs.join(thread["thread_id"], run["run_id"])

# %%
# Eventually, it should finish and we should see `status=success`
await client.runs.get(thread["thread_id"], run["run_id"])

# %%
# We can get the final results
final_result = await client.threads.get_state(thread["thread_id"])

# %%
final_result

# %%
# We can get the content of the final message
final_result["values"]["messages"][-1]["content"]

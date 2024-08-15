# %% [markdown]
# # Cron Jobs
#
# Sometimes you don't want to run your graph based on user interaction, but rather you would like to schedule your graph to run on a schedule - for example if you wish for your graph to compose and send out a weekly email of to-dos for your team. LangGraph Cloud allows you to do this without having to write your own script by using the `Crons` client. To schedule a graph job, you need to pass a [cron expression](https://crontab.cronhub.io/) to inform the client when you want to run the graph. `Cron` jobs are run in the background and do not interfere with normal invocations of the graph.
#
# ## Setup
#
# First, let's setup our SDK client, assistant, and thread:

# %%
from langgraph_sdk import get_client

client = get_client()
assistants = await client.assistants.search()
assistants = [a for a in assistants if not a["config"]]
assistant = assistants[0]
thread = await client.threads.create()

# %% [markdown]
# ## Cron job on a thread 
#
# To create a cron job associated with a specific thread, you can write:

# %%
# This schedules a job to run at 15:27 (3:27PM) every day
cron_1 = await client.crons.create_for_thread(
    thread["thread_id"],
    assistant["assistant_id"],
    schedule="27 15 * * *",
    input={"messages": [{"role": "user", "content": "What time is it?"}]},
)

# %% [markdown]
# Note that it is **very** important to delete `Cron` jobs that are no longer useful. Otherwise you could rack up unwanted API charges to the LLM! You can delete a `Cron` job using the following code:

# %%
await client.crons.delete(cron_1["cron_id"])

# %% [markdown]
# ## Cron job stateless
#
# You can also create stateless cron jobs by using the following code:

# %%
# This schedules a job to run at 15:27 (3:27PM) every day
cron_2 = await client.crons.create(
    assistant["assistant_id"],
    schedule="27 15 * * *",
    input={"messages": [{"role": "user", "content": "What time is it?"}]},
)

# %% [markdown]
# Again, remember to delete your job once you are done with it!

# %%
await client.crons.delete(cron_2["cron_id"])

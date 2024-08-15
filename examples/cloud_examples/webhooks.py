# %% [markdown]
# # Use Webhooks
#
# You may wish to use webhooks in your client, especially when using async streams in case you want to update something in your service once the API call to LangGraph Cloud has finished running. To do so, you will need to expose an endpoint that can accept POST requests, and then pass it to your API request in the "webhook" parameter.
#
# Currently, the SDK has not exposed this endpoint but you can access it through curl commands as follows.
#
# The following endpoints accept `webhook` as a parameter: 
#
# - Create Run -> POST /thread/{thread_id}/runs
# - Create Thread Cron -> POST /thread/{thread_id}/runs/crons
# - Stream Run -> POST /thread/{thread_id}/runs/stream
# - Wait Run -> POST /thread/{thread_id}/runs/wait
# - Create Cron -> POST /runs/crons
# - Stream Run Stateless -> POST /runs/stream
# - Wait Run Stateless -> POST /runs/wait
#
# The following example uses a url from a public website that allows users to create free webhooks, but you should pass in the webhook that you wish to use. 

# %%
curl --request POST \
  --url http://localhost:8123/threads/b76d1e94-f251-40e3-8933-796d775cdb4c/runs/stream \
  --header 'Content-Type: application/json' \
  --data '{
  "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
  "input" : {"messages":[{"role": "user", "content": "Hello! My name is Bagatur and I am 26 years old."}]},
  "metadata": {},
  "config": {
    "configurable": {}
  },
  "multitask_strategy": "reject",
  "stream_mode": [
    "values"
  ],
 "webhook": "https://webhook.site/6ca33471-dd65-4103-a851-0a252dae0f2a"
}'

# %% [markdown]
# To check that this worked as intended, we can go to the website where our webhook was created and confirm that it received a POST request:

# %% [markdown]
# ![Webhook response](./img/webhook_results.png)

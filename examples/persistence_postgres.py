# %% [markdown]
# # How to use Postgres checkpointer for persistence
#
# When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple times and have it remember previous interactions.
#
# This example shows how to use `Postgres` as the backend for persisting checkpoint state using [`langgraph-checkpoint-postgres`](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres) library.
#
# To start a Postgres database to work with you can do the following:
#
# ```
# $ cd libs/langgraph
# $ make start-postgres

# %% [markdown]
# ## Setup environment

# %%
# %%capture --no-stderr
# %pip install -U psycopg psycopg-pool langgraph langgraph-checkpoint-postgres

# %%
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")

# %% [markdown]
# ## Setup model and tools for the graph

# %%
from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# %% [markdown]
# ## Use sync connection
#
# This sets up a synchronous connection to the database. 
#
# Synchronous connections execute operations in a blocking manner, meaning each operation waits for completion before moving to the next one. The `DB_URI` is the database connection URI, with the protocol used for connecting to a PostgreSQL database, authentication, and host where database is running. The connection_kwargs dictionary defines additional parameters for the database connection.

# %%
DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

# %%
from psycopg.rows import dict_row

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# %% [markdown]
# ### With a connection pool
#
# This manages a pool of reusable database connections: 
# - Advantages: Efficient resource utilization, improved performance for frequent connections
# - Best for: Applications with many short-lived database operations
#

# %%
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    # Example configuration
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
)

with pool.connection() as conn:
    checkpointer = PostgresSaver(conn)

    # NOTE: you need to call .setup() the first time you're using your checkpointer
    checkpointer.setup()

    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)
    checkpoint = checkpointer.get(config)

# %%
res

# %%
checkpoint

# %% [markdown]
# ### With a connection
#
# This creates a single, dedicated connection to the database:
# - Advantages: Simple to use, suitable for longer transactions
# - Best for: Applications with fewer, longer-lived database operations

# %%
from psycopg import Connection


with Connection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer = PostgresSaver(conn)
    # NOTE: you need to call .setup() the first time you're using your checkpointer
    # checkpointer.setup()
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "2"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    checkpoint_tuple = checkpointer.get_tuple(config)

# %%
checkpoint_tuple

# %% [markdown]
# ### With a connection string
#
# This creates a connection based on a connection string:
# - Advantages: Simplicity, encapsulates connection details
# - Best for: Quick setup or when connection details are provided as a string

# %%
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "3"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    checkpoint_tuples = list(checkpointer.list(config))

# %%
checkpoint_tuples

# %% [markdown]
# ## Use async connection
#
# This sets up an asynchronous connection to the database. 
#
# Async connections allow non-blocking database operations. This means other parts of your application can continue running while waiting for database operations to complete. It's particularly useful in high-concurrency scenarios or when dealing with I/O-bound operations.

# %% [markdown]
# ### With a connection pool

# %%
from psycopg_pool import AsyncConnectionPool

async with AsyncConnectionPool(
    # Example configuration
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
) as pool, pool.connection() as conn:
    checkpointer = AsyncPostgresSaver(conn)

    # NOTE: you need to call .setup() the first time you're using your checkpointer
    # await checkpointer.setup()

    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "4"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )

    checkpoint = await checkpointer.aget(config)

# %%
checkpoint

# %% [markdown]
# ### With a connection

# %%
from psycopg import AsyncConnection

async with await AsyncConnection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer = AsyncPostgresSaver(conn)
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "5"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )
    checkpoint_tuple = await checkpointer.aget_tuple(config)

# %%
checkpoint_tuple

# %% [markdown]
# ### With a connection string

# %%
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "6"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )
    checkpoint_tuples = [c async for c in checkpointer.alist(config)]

# %%
checkpoint_tuples

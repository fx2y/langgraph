# %% [markdown]
# # An agent for interacting with a SQL database
#
# In this tutorial, we will walk through how to build an agent that can answer questions about a SQL database. 
#
# At a high level, the agent will:
# 1. Fetch the available tables from the database
# 2. Decide which tables are relevant to the question
# 3. Fetch the DDL for the relevant tables
# 4. Generate a query based on the question and information from the DDL
# 5. Double-check the query for common mistakes using an LLM
# 6. Execute the query and return the results
# 7. Correct mistakes surfaced by the database engine until the query is successful
# 8. Formulate a response based on the results
#
# The end-to-end workflow will look something like below:
#
# ![](sql-agent-diagram.png)

# %% [markdown]
# ## Set up environment
#
# We'll set up our environment variables for OpenAI, and optionally, to enable tracing with [LangSmith](https://smith.langchain.com).

# %%
import os

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# %%
os.environ["LANGCHAIN_PROJECT"] = "sql-agent"

# %% [markdown]
# ## Configure the database
#
# We will be creating a SQLite database for this tutorial. SQLite is a lightweight database that is easy to set up and use. We will be loading the `chinook` database, which is a sample database that represents a digital media store.
# Find more information about the database [here](https://www.sqlitetutorial.net/sqlite-sample-database/).
#
# For convenience, we have hosted the database (`Chinook.db`) on a public GCS bucket.

# %%
import requests

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

response = requests.get(url)

if response.status_code == 200:
    # Open a local file in binary write mode
    with open("Chinook.db", "wb") as file:
        # Write the content of the response (the file) to the local file
        file.write(response.content)
    print("File downloaded and saved as Chinook.db")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

# %% [markdown]
# We will use a handy SQL database wrapper available in the `langchain_community` package to interact with the database. The wrapper provides a simple interface to execute SQL queries and fetch results. We will also use the `langchain_openai` package to interact with the OpenAI API for language models later in the tutorial.

# %%
# %%capture --no-stderr --no-display
# !pip install langgraph langchain_community langchain_openai

# %%
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM Artist LIMIT 10;")

# %% [markdown]
# ## Utility functions
#
# We will define a few utility functions to help us with the agent implementation. Specifically, we will wrap a `ToolNode` with a fallback to handle errors and surface them to the agent.

# %%
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# %% [markdown]
# ## Define tools for the agent
#
# We will define a few tools that the agent will use to interact with the database.
#
# 1. `list_tables_tool`: Fetch the available tables from the database
# 2. `get_schema_tool`: Fetch the DDL for a table
# 3. `db_query_tool`: Execute the query and fetch the results OR return an error message if the query fails
#
# For the first two tools, we will grab them from the `SQLDatabaseToolkit`, also available in the `langchain_community` package.

# %%
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model="gpt-4o"))
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

print(list_tables_tool.invoke(""))

print(get_schema_tool.invoke("Artist"))

# %% [markdown]
# The third will be defined manually. For the `db_query_tool`, we will execute the query against the database and return the results.

# %%
from langchain_core.tools import tool


@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result


print(db_query_tool.invoke("SELECT * FROM Artist LIMIT 10;"))

# %% [markdown]
# While not strictly a tool, we will prompt an LLM to check for common mistakes in the query and later add this as a node in the workflow.

# %%
from langchain_core.prompts import ChatPromptTemplate

query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)
query_check = query_check_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [db_query_tool], tool_choice="required"
)

query_check.invoke({"messages": [("user", "SELECT * FROM Artist LIMIT 10;")]})

# %% [markdown]
# ## Define the workflow
#
# We will then define the workflow for the agent. The agent will first force-call the `list_tables_tool` to fetch the available tables from the database, then follow the steps mentioned at the beginning of the tutorial.

# %%
from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages


# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Define a new graph
workflow = StateGraph(State)


# Add a node for the first tool call
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}


workflow.add_node("first_tool_call", first_tool_call)

# Add nodes for the first two tools
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

# Add a node for a model to choose the relevant tables based on the question and available tables
model_get_schema = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [get_schema_tool]
)
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)


# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""

    final_answer: str = Field(..., description="The final answer to the user")


# Add a node for a model to generate a query based on the question and schema
query_gen_system = """You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

When generating the query:

Output the SQL query that answers the input question without a tool call.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

If you get an error while executing a query, rewrite the query and try again.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
    [SubmitFinalAnswer]
)


def query_gen_node(state: State):
    message = query_gen.invoke(state)

    # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}


workflow.add_node("query_gen", query_gen_node)

# Add a node for the model to check the query before executing it
workflow.add_node("correct_query", model_check_query)

# Add node for executing the query
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))


# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"


# Specify the edges between the nodes
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges(
    "query_gen",
    should_continue,
)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")

# Compile the workflow into a runnable
app = workflow.compile()

# %% [markdown]
# ## Visualize the graph

# %%
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)

# %% [markdown]
# ## Run the agent

# %%
import json

messages = app.invoke(
    {"messages": [("user", "Which sales agent made the most in sales in 2009?")]}
)
json_str = messages["messages"][-1].additional_kwargs["tool_calls"][0]["function"][
    "arguments"
]
json.loads(json_str)["final_answer"]

# %%
for event in app.stream(
    {"messages": [("user", "Which sales agent made the most in sales in 2009?")]}
):
    print(event)

# %% [markdown]
# ## Eval
#
# Now, we can evaluate this agent! We previously defined [simple SQL agent](https://github.com/langchain-ai/langsmith-cookbook/blob/main/testing-examples/agent-evals-with-langgraph/langgraph_sql_agent_eval.ipynb) as part of our LangSmith evaluation cookbooks, and evaluated responses to 5 questions about our database. We can compare this agent to our prior one on the same dataset. [Agent evaluation](https://docs.smith.langchain.com/concepts/evaluation#agents) can focus on 3 things:
#
# * `Response`: The inputs are a prompt and a list of tools. The output is the agent response.
# * `Single tool`: As before, the inputs are a prompt and a list of tools. The output the tool call.
# * `Trajectory`: As before, the inputs are a prompt and a list of tools. The output is the list of tool calls
#
# ![Screenshot 2024-06-13 at 2.13.30 PM.png](attachment:b92325b1-2c9a-4efa-94f5-49a75b1ffb64.png)
#
# ### Response
#
# We'll evaluate end-to-end responses of our agent relative to reference answers. Let's run [response evaluation](https://docs.smith.langchain.com/concepts/evaluation#evaluating-an-agents-final-response) [on the same dataset](https://smith.langchain.com/public/20808486-67c3-4e30-920b-6d49d6f2b6b8/d).

# %%
import json


def predict_sql_agent_answer(example: dict):
    """Use this for answer evaluation"""
    msg = {"messages": ("user", example["input"])}
    messages = app.invoke(msg)
    json_str = messages["messages"][-1].additional_kwargs["tool_calls"][0]["function"][
        "arguments"
    ]
    response = json.loads(json_str)["final_answer"]
    return {"response": response}


# %%
from langchain import hub
from langchain_openai import ChatOpenAI

# Grade prompt
grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")


def answer_evaluator(run, example) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """

    # Get question, ground truth answer, chain
    input_question = example.inputs["input"]
    reference = example.outputs["output"]
    prediction = run.outputs["response"]

    # LLM grader
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # Run evaluator
    score = answer_grader.invoke(
        {
            "question": input_question,
            "correct_answer": reference,
            "student_answer": prediction,
        }
    )
    score = score["Score"]

    return {"key": "answer_v_reference_score", "score": score}


# %%
from langsmith.evaluation import evaluate

dataset_name = "SQL Agent Response"
experiment_results = evaluate(
    predict_sql_agent_answer,
    data=dataset_name,
    evaluators=[answer_evaluator],
    num_repetitions=3,
    experiment_prefix="sql-agent-multi-step-response-v-reference",
    metadata={"version": "Chinook, gpt-4o multi-step-agent"},
)

# %% [markdown]
# Summary metrics (see dataset [here](https://smith.langchain.com/public/20808486-67c3-4e30-920b-6d49d6f2b6b8/d)):
#
# * The `multi-step` agent here out performs the previously defined [base case SQL agent](https://github.com/langchain-ai/langsmith-cookbook/blob/main/testing-examples/agent-evals-with-langgraph/langgraph_sql_agent_eval.ipynb)
#
# ![Screenshot 2024-06-13 at 2.09.57 PM.png](attachment:e9a91890-3299-4b71-9ab2-21d737a120ba.png)

# %% [markdown]
# ### Trajectory
#
# Let's run [trajectory evaluation](https://docs.smith.langchain.com/concepts/evaluation#evaluating-an-agents-trajectory) on this same dataset.

# %%
# These are the tools that we expect the agent to use
expected_trajectory = [
    "sql_db_list_tables",  # first: list_tables_tool node
    "sql_db_schema",  # second: get_schema_tool node
    "db_query_tool",  # third: execute_query node
    "SubmitFinalAnswer",
]  # fourth: query_gen


# %%
def predict_sql_agent_messages(example: dict):
    """Use this for answer evaluation"""
    msg = {"messages": ("user", example["input"])}
    messages = app.invoke(msg)
    return {"response": messages}


# %%
from langsmith.schemas import Example, Run


def find_tool_calls(messages):
    """
    Find all tool calls in the messages returned
    """
    tool_calls = [
        tc["name"] for m in messages["messages"] for tc in getattr(m, "tool_calls", [])
    ]
    return tool_calls


def contains_all_tool_calls_in_order_exact_match(
    root_run: Run, example: Example
) -> dict:
    """
    Check if all expected tools are called in exact order and without any additional tool calls.
    """
    expected_trajectory = [
        "sql_db_list_tables",
        "sql_db_schema",
        "db_query_tool",
        "SubmitFinalAnswer",
    ]
    messages = root_run.outputs["response"]
    tool_calls = find_tool_calls(messages)

    # Print the tool calls for debugging
    print("Here are my tool calls:")
    print(tool_calls)

    # Check if the tool calls match the expected trajectory exactly
    if tool_calls == expected_trajectory:
        score = 1
    else:
        score = 0

    return {"score": int(score), "key": "multi_tool_call_in_exact_order"}


def contains_all_tool_calls_in_order(root_run: Run, example: Example) -> dict:
    """
    Check if all expected tools are called in order,
    but it allows for other tools to be called in between the expected ones.
    """
    messages = root_run.outputs["response"]
    tool_calls = find_tool_calls(messages)

    # Print the tool calls for debugging
    print("Here are my tool calls:")
    print(tool_calls)

    it = iter(tool_calls)
    if all(elem in it for elem in expected_trajectory):
        score = 1
    else:
        score = 0
    return {"score": int(score), "key": "multi_tool_call_in_order"}


# %%
experiment_results = evaluate(
    predict_sql_agent_messages,
    data=dataset_name,
    evaluators=[
        contains_all_tool_calls_in_order,
        contains_all_tool_calls_in_order_exact_match,
    ],
    num_repetitions=3,
    experiment_prefix="sql-agent-multi-step-tool-calling-trajecory-in-order",
    metadata={"version": "Chinook, gpt-4o multi-step-agent"},
)

# %% [markdown]
# The aggregate scores show that we never correctly call the tools in exact order:
#
# ![Screenshot 2024-06-13 at 2.46.34 PM.png](attachment:9a1084c0-4c7c-4e6f-8329-80499d293e0a.png)

# %% [markdown]
# Looking at the logging, we can see something interesting - 
#
# ```
# ['sql_db_list_tables', 'sql_db_schema', 'sql_db_query', 'db_query_tool', 'SubmitFinalAnswer']
# ```
#
# We appear to inject a hallucinated tool call, `sql_db_query`, into our trajectory for most of the runs.
#
# This is why `multi_tool_call_in_exact_order` fails, but `multi_tool_call_in_order` still passes. 
#
# We will explore ways to resolve this using LangGraph in future cookbooks!

# %%

# %% [markdown]
# # How to pass private state
#
# Oftentimes, you may want nodes to be able to pass state to each other that should NOT be part of the main schema of the graph. This is often useful because there may be information that is not needed as input/output (and therefore doesn't really make sense to have in the main schema) but is ABSOLUTELY needed as part of the intermediate working logic.
#
# Let's take a look at an example below. In this example, we will create a RAG pipeline that:
# 1. Takes in a user question
# 2. Uses an LLM to generate a search query
# 3. Retrieves documents for that generated query
# 4. Generates a final answer based on those documents
#
# We will have a separate node for each step. We will only have the `question` and `answer` on the overall state. However, we will need separate states for the `search_query` and the `documents` - we will pass these as private state keys.
#
# Let's look at an example!

# %%
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# The overall state of the graph
class OverallState(TypedDict):
    question: str
    answer: str


# This is what the node that generates the query will return
class QueryOutputState(TypedDict):
    query: str


# This is what the node that retrieves the documents will return
class DocumentOutputState(TypedDict):
    docs: list[str]


# This is what the node that generates the final answer will take in
class GenerateInputState(OverallState, DocumentOutputState):
    pass


# Node to generate query
def generate_query(state: OverallState) -> QueryOutputState:
    # Replace this with real logic
    return {"query": state["question"][:2]}


# Node to retrieve documents
def retrieve_documents(state: QueryOutputState) -> DocumentOutputState:
    # Replace this with real logic
    return {"docs": [state['query']] * 2}


# Node to generate answer
def generate(state: GenerateInputState) -> OverallState:
    return {"answer": "\n\n".join(state['docs'] + [state['question']])}


graph = StateGraph(OverallState)
graph.add_node(generate_query)
graph.add_node(retrieve_documents)
graph.add_node(generate)
graph.add_edge(START, "generate_query")
graph.add_edge("generate_query", "retrieve_documents")
graph.add_edge("retrieve_documents", "generate")
graph.add_edge("generate", END)
graph = graph.compile()

graph.invoke({"question": "foo"})

# %%

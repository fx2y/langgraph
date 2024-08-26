import json
import os
import re
import shutil
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]

examples_dir = root_dir / "examples"
docs_dir = root_dir / "docs/docs"
how_tos_dir = docs_dir / "how-tos"
tutorials_dir = docs_dir / "tutorials"
cloud_how_tos_dir = docs_dir / "cloud/how-tos"
cloud_sdk_dir = docs_dir / "cloud"

_MANUAL = {
    "how-tos": [
        "state-context-key.ipynb",
        "async.ipynb",
        "stream-values.ipynb",
        "stream-updates.ipynb",
        "stream-multiple.ipynb",
        "streaming-tokens.ipynb",
        "streaming-tokens-without-langchain.ipynb",
        "streaming-content.ipynb",
        "streaming-events-from-within-tools.ipynb",
        "streaming-events-from-within-tools-without-langchain.ipynb",
        "streaming-from-final-node.ipynb",
        "persistence.ipynb",
        "input_output_schema.ipynb",
        "pass_private_state.ipynb",
        "memory/manage-conversation-history.ipynb",
        "memory/delete-messages.ipynb",
        "memory/add-summary-conversation-history.ipynb",
        "persistence_postgres.ipynb",
        "persistence_mongodb.ipynb",
        "persistence_redis.ipynb",
        "visualization.ipynb",
        "state-model.ipynb",
        "subgraph.ipynb",
        "force-calling-a-tool-first.ipynb",
        "pass-run-time-values-to-tools.ipynb",
        "tool-calling.ipynb",
        "tool-calling-errors.ipynb",
        "pass-config-to-tools.ipynb",
        "many-tools.ipynb",
        "dynamic-returning-direct.ipynb",
        "managing-agent-steps.ipynb",
        "respond-in-format.ipynb",
        "branching.ipynb",
        "dynamically-returning-directly.ipynb",
        "configuration.ipynb",
        "map-reduce.ipynb",
        "create-react-agent.ipynb",
        "create-react-agent-system-prompt.ipynb",
        "create-react-agent-memory.ipynb",
        "create-react-agent-hitl.ipynb",
        "human_in_the_loop/breakpoints.ipynb",
        "human_in_the_loop/dynamic_breakpoints.ipynb",
        "human_in_the_loop/time-travel.ipynb",
        "human_in_the_loop/edit-graph-state.ipynb",
        "human_in_the_loop/wait-user-input.ipynb",
        "human_in_the_loop/review-tool-calls.ipynb",
        "node-retries.ipynb",
    ],
    "tutorials": [
        "introduction.ipynb",
        "customer-support/customer-support.ipynb",
        "tutorials/tnt-llm/tnt-llm.ipynb",
        "tutorials/sql-agent.ipynb",
    ],
}
_MANUAL_INVERSE = {v: docs_dir / k for k, vs in _MANUAL.items() for v in vs}
_HOW_TOS = {"agent_executor", "chat_agent_executor_with_function_calling", "docs"}
_HIDE = set(
    str(examples_dir / f)
    for f in [
        "agent_executor/base.ipynb",
        "agent_executor/force-calling-a-tool-first.ipynb",
        "agent_executor/high-level.ipynb",
        "agent_executor/human-in-the-loop.ipynb",
        "agent_executor/managing-agent-steps.ipynb",
        "chat_agent_executor_with_function_calling/anthropic.ipynb",
        "chat_agent_executor_with_function_calling/base.ipynb",
        "chat_agent_executor_with_function_calling/dynamically-returning-directly.ipynb",
        "chat_agent_executor_with_function_calling/force-calling-a-tool-first.ipynb",
        "chat_agent_executor_with_function_calling/high-level-tools.ipynb",
        "chat_agent_executor_with_function_calling/high-level.ipynb",
        "chat_agent_executor_with_function_calling/human-in-the-loop.ipynb",
        "chat_agent_executor_with_function_calling/managing-agent-steps.ipynb",
        "chat_agent_executor_with_function_calling/prebuilt-tool-node.ipynb",
        "chat_agent_executor_with_function_calling/respond-in-format.ipynb",
        "chatbots/customer-support.ipynb",
        "rag/langgraph_rag_agent_llama3_local.ipynb",
        "rag/langgraph_self_rag_pinecone_movies.ipynb",
        "rag/langgraph_adaptive_rag_cohere.ipynb",
        "dynamically-returning-directly.ipynb",
        "force-calling-a-tool-first.ipynb",
        "managing-agent-steps.ipynb",
        "respond-in-format.ipynb",
        "quickstart.ipynb",
        "human-in-the-loop.ipynb",
        "learning.ipynb",
        "docs/quickstart.ipynb",
        "tutorials/rag-agent-testing.ipynb",
        "tutorials/rag-agent-testing-local.ipynb",
        "tutorials/tool-calling-agent-local.ipynb",
        "time-travel.ipynb",
        "code_assistant/langgraph_code_assistant_mistral.ipynb",
    ]
)


def clean_notebooks():
    roots = (how_tos_dir, tutorials_dir)
    for dir_ in roots:
        traversed = []
        for root, dirs, files in os.walk(dir_):
            for file in files:
                if file.endswith(".ipynb"):
                    os.remove(os.path.join(root, file))
            # Now delete the dir if it is empty now
            if root not in roots:
                traversed.append(root)

        for root in reversed(traversed):
            if not os.listdir(root):
                os.rmdir(root)


def update_notebook_links(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            for i, source in enumerate(cell["source"]):
                # Update relative notebook links
                cell["source"][i] = re.sub(
                    r"\[([^\]]+)\]\(([^:)]+\.ipynb)\)",
                    lambda m: transform_link(m.group(1), m.group(2)),
                    source,
                )

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)


def transform_link(text, link):
    dir_path, filename = os.path.split(link)

    # Remove the .ipynb extension
    filename_without_ext = os.path.splitext(filename)[0]

    # If it's a local link (starts with ./)
    if link.startswith("./"):
        # Change to parent directory and remove ./ prefix
        new_link = f"../{filename_without_ext}/"
    elif dir_path:
        # If there's a directory path, keep it and add one more level up
        new_link = f"../{dir_path}/{filename_without_ext}/"
    else:
        # If it's just a filename, simply go one level up
        new_link = f"../{filename_without_ext}/"

    return f"[{text}]({new_link})"


def copy_notebooks():
    # Nested ones are mostly tutorials rn
    for root, dirs, files in os.walk(examples_dir):
        if any(
            path.startswith(".") or path.startswith("__") for path in root.split(os.sep)
        ):
            continue
        if any(path in _HOW_TOS for path in root.split(os.sep)):
            dst_dir = how_tos_dir
        elif "sdk" in root.split(os.sep):
            dst_dir = cloud_sdk_dir
        elif "cloud_examples" in root.split(os.sep):
            dst_dir = cloud_how_tos_dir
        else:
            dst_dir = tutorials_dir
        for file in files:
            dst_dir_ = dst_dir
            if file.endswith((".ipynb", ".png")):
                src_path = os.path.join(root, file)
                if src_path in _HIDE:
                    print("Hiding:", src_path)
                    continue
                dst_path = os.path.join(
                    dst_dir, os.path.relpath(src_path, examples_dir)
                )
                for k in _MANUAL_INVERSE:
                    if src_path.endswith(k):
                        overridden_dir = _MANUAL_INVERSE[k]
                        dst_path = os.path.join(
                            overridden_dir, os.path.relpath(src_path, examples_dir)
                        )
                        print(f"Overriding: {src_path} to {dst_path}")
                        break
                # Avoid double nesting.
                dst_path = dst_path.replace("tutorials/tutorials", "tutorials").replace(
                    "how-tos/how-tos", "how-tos"
                )
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                print(f"Copying: {src_path} to {dst_path}")
                shutil.copy(src_path, dst_path)
                # Convert all ./img/* to ../img/*
                if file.endswith(".ipynb"):
                    with open(dst_path, "r") as f:
                        content = f.read()
                    content = content.replace("(./img/", "(../img/")
                    content = content.replace('src=\\"./img/', 'src=\\"../img/')
                    with open(dst_path, "w") as f:
                        f.write(content)
                    update_notebook_links(dst_path)
                dst_dir = dst_dir_


if __name__ == "__main__":
    clean_notebooks()
    copy_notebooks()

import re
import time
import logging
from typing import List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableBranch, chain as as_runnable
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Task(BaseModel):
    idx: int
    tool: Union[BaseTool, str]
    args: Union[str, Dict[str, Any]]
    dependencies: List[int]
    priority: int = 0
    estimated_duration: float = 1.0  # Estimated task duration in seconds

class LLMCompilerAgent:
    def __init__(self, llm: BaseChatModel, tools: List[BaseTool], planner_prompt: ChatPromptTemplate):
        self.llm = llm
        self.tools = tools
        self.planner = self._create_planner(planner_prompt)
        self.joiner = self._create_joiner()
        logger.info("LLMCompilerAgent initialized with %d tools", len(tools))
        self.metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "replans": 0,
            "execution_time": 0,
            "average_task_duration": 0,
            "task_success_rate": 0,
            "plan_efficiency": 0,
        }
        self.error_threshold = 3  # Maximum number of consecutive errors before recovery

    def _create_planner(self, base_prompt: ChatPromptTemplate):
        tool_descriptions = "\n".join(
            f"{i+1}. {tool.description}\n" for i, tool in enumerate(self.tools)
        )
        planner_prompt = base_prompt.partial(
            replan="",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
        )
        replanner_prompt = base_prompt.partial(
            replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
            "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
            'You MUST use these information to create the next plan under "Current Plan".\n'
            ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
            " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
            " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
        )

        def should_replan(state: list):
            return isinstance(state[-1], SystemMessage)

        def wrap_messages(state: list):
            return {"messages": state}

        def wrap_and_get_last_index(state: list):
            next_task = 0
            for message in state[::-1]:
                if isinstance(message, FunctionMessage):
                    next_task = message.additional_kwargs["idx"] + 1
                    break
            state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
            return {"messages": state}

        def variable_substitution(tasks: List[Task]) -> List[Task]:
            for task in tasks:
                if isinstance(task.args, str):
                    task.args = self._resolve_arg(task.args, {t.idx: None for t in tasks})
                elif isinstance(task.args, dict):
                    task.args = {k: self._resolve_arg(v, {t.idx: None for t in tasks}) for k, v in task.args.items()}
            return tasks

        def task_generation(llm_output: str) -> List[Task]:
            # Implement more sophisticated task generation logic
            # This could include task prioritization, dependency analysis, etc.
            tasks = LLMCompilerPlanParser(tools=self.tools).parse(llm_output)
            for task in tasks:
                task.priority = self._calculate_task_priority(task)
            return tasks

        return (
            RunnableBranch(
                (should_replan, wrap_and_get_last_index | replanner_prompt),
                wrap_messages | planner_prompt,
            )
            | self.llm
            | task_generation
            | variable_substitution
        )

    def _calculate_task_priority(self, task: Task) -> int:
        priority = 0
        
        # Prioritize tasks with more dependencies
        priority += len(task.dependencies) * 2
        
        # Prioritize certain tools
        tool_priorities = {
            "web_search": 5,
            "database_query": 4,
            "file_operation": 3,
            "calculation": 2,
        }
        priority += tool_priorities.get(str(task.tool), 0)
        
        # Prioritize tasks with shorter estimated duration
        priority += max(10 - int(task.estimated_duration), 0)
        
        # Deprioritize tasks with potential security risks
        if any(risk_keyword in str(task.args) for risk_keyword in ["delete", "remove", "drop"]):
            priority -= 5
        
        return priority

    def _create_joiner(self):
        class FinalResponse(BaseModel):
            response: str

        class Replan(BaseModel):
            feedback: str = Field(
                description="Analysis of the previous attempts and recommendations on what needs to be fixed."
            )

        class JoinOutputs(BaseModel):
            thought: str = Field(
                description="The chain of thought reasoning for the selected action"
            )
            action: Union[FinalResponse, Replan]

        joiner_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a decision-making component that determines whether to provide a final response or request a replan."),
            HumanMessage(content="Based on the following conversation and task results, decide whether to provide a final response or request a replan:\n\n{conversation}\n\nProvide your decision in the specified format."),
        ])

        runnable = create_structured_output_runnable(JoinOutputs, self.llm, joiner_prompt)

        def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
            response = [AIMessage(content=f"Thought: {decision.thought}")]
            if isinstance(decision.action, Replan):
                return response + [
                    SystemMessage(
                        content=f"Context from last attempt: {decision.action.feedback}"
                    )
                ]
            else:
                return {"messages": response + [AIMessage(content=decision.action.response)]}

        def select_recent_messages(state) -> dict:
            messages = state["messages"]
            selected = []
            for msg in messages[::-1]:
                selected.append(msg)
                if isinstance(msg, HumanMessage):
                    break
            return {"messages": selected[::-1]}

        def decision_making(join_output: JoinOutputs) -> Dict[str, Any]:
            if isinstance(join_output.action, FinalResponse):
                return {"messages": [AIMessage(content=join_output.action.response)], "decision": "final_response"}
            elif isinstance(join_output.action, Replan):
                return {"messages": [SystemMessage(content=f"Replan needed: {join_output.action.feedback}")], "decision": "replan"}
            else:
                raise ValueError("Invalid action type in JoinOutputs")

        def replanning(state: Dict[str, Any]) -> List[BaseMessage]:
            if state["decision"] == "replan":
                replan_message = state["messages"][-1].content
                logger.info(f"Replanning initiated: {replan_message}")
                return [SystemMessage(content=replan_message)]
            else:
                return state["messages"]

        return select_recent_messages | runnable | decision_making | replanning

    def _execute_task(self, task: Task, observations: Dict[int, Any]):
        tool_to_use = task.tool
        try:
            args = self._resolve_args(task.args, observations)
            logger.info(f"Executing task {task.idx} with tool {tool_to_use}")
            result = tool_to_use.invoke(args)
            observations[task.idx] = result
            logger.info(f"Task {task.idx} completed successfully")
        except Exception as e:
            error_message = f"ERROR: {str(e)}"
            observations[task.idx] = error_message
            logger.error(f"Error executing task {task.idx}: {error_message}", exc_info=True)

    def _resolve_args(self, args: Union[str, Dict[str, Any]], observations: Dict[int, Any]) -> Any:
        try:
            if isinstance(args, str):
                return self._resolve_arg(args, observations)
            elif isinstance(args, dict):
                return {key: self._resolve_arg(val, observations) for key, val in args.items()}
            else:
                return args
        except Exception as e:
            logger.error(f"Error resolving arguments: {str(e)}", exc_info=True)
            raise

    def _resolve_arg(self, arg: Union[str, Any], observations: Dict[int, Any]) -> Any:
        try:
            if isinstance(arg, str):
                return re.sub(r'\$\{?(\d+)\}?', lambda m: str(observations.get(int(m.group(1)), m.group(0))), arg)
            elif isinstance(arg, list):
                return [self._resolve_arg(a, observations) for a in arg]
            else:
                return str(arg)
        except Exception as e:
            logger.error(f"Error resolving argument {arg}: {str(e)}", exc_info=True)
            raise

    @as_runnable
    def execute_tasks(self, scheduler_input: Dict[str, Any]) -> List[FunctionMessage]:
        tasks = scheduler_input["tasks"]
        messages = scheduler_input["messages"]
        observations = self._get_observations(messages)
        
        def task_scheduling():
            scheduled_tasks = []
            pending_tasks = sorted(tasks, key=lambda t: (t.priority, -t.estimated_duration), reverse=True)
            total_duration = sum(task.estimated_duration for task in pending_tasks)
            current_time = 0
            
            while pending_tasks and current_time < total_duration * 1.5:  # Allow 50% overtime
                executable_tasks = [
                    task for task in pending_tasks
                    if all(dep in observations for dep in task.dependencies)
                ]
                
                if executable_tasks:
                    task = max(executable_tasks, key=lambda t: t.priority / t.estimated_duration)
                    scheduled_tasks.append(task)
                    pending_tasks.remove(task)
                    current_time += task.estimated_duration
                else:
                    current_time += 0.1  # Wait for dependencies to be met
            
            return scheduled_tasks

        scheduled_tasks = task_scheduling()
        
        start_time = time.time()
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._execute_task, task, observations) for task in scheduled_tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                    self.metrics["successful_tasks"] += 1
                except Exception as e:
                    self.metrics["failed_tasks"] += 1
                    logger.error(f"Error in task execution: {str(e)}", exc_info=True)
        
        execution_time = time.time() - start_time
        self.metrics["execution_time"] += execution_time
        self.metrics["total_tasks"] += len(scheduled_tasks)
        self.metrics["average_task_duration"] = execution_time / len(scheduled_tasks) if scheduled_tasks else 0
        self.metrics["task_success_rate"] = self.metrics["successful_tasks"] / self.metrics["total_tasks"] if self.metrics["total_tasks"] > 0 else 0
        self.metrics["plan_efficiency"] = len(scheduled_tasks) / len(tasks) if tasks else 1

        return self._create_tool_messages(observations)

    def _get_observations(self, messages: List[BaseMessage]) -> Dict[int, Any]:
        results = {}
        for message in messages[::-1]:
            if isinstance(message, FunctionMessage):
                results[int(message.additional_kwargs["idx"])] = message.content
        return results

    def _create_tool_messages(self, observations: Dict[int, Any]) -> List[FunctionMessage]:
        return [
            FunctionMessage(
                name=str(task.tool),
                content=str(result),
                additional_kwargs={"idx": idx, "args": task.args}
            )
            for idx, (task, result) in observations.items()
        ]

    def run(self, input_message: str) -> str:
        state = {"messages": [HumanMessage(content=input_message)]}
        
        max_iterations = 10
        consecutive_errors = 0
        
        for iteration in range(max_iterations):
            logger.info(f"Starting iteration {iteration + 1}")
            
            try:
                tasks = self.planner.invoke(state["messages"])
                tool_messages = self.execute_tasks({"tasks": tasks, "messages": state["messages"]})
                state["messages"].extend(tool_messages)

                result = self.joiner.invoke(state)
                state["messages"].extend(result["messages"])

                if result["decision"] == "final_response":
                    logger.info("Final response generated")
                    self._log_metrics()
                    return result["messages"][-1].content
                elif result["decision"] == "replan":
                    logger.info("Replanning initiated")
                    self.metrics["replans"] += 1
                    consecutive_errors = 0  # Reset error count on successful replan
                    continue
            except Exception as e:
                logger.error(f"Error in run iteration {iteration + 1}: {str(e)}", exc_info=True)
                consecutive_errors += 1
                
                if consecutive_errors >= self.error_threshold:
                    recovery_message = self._error_recovery(state, consecutive_errors)
                    state["messages"].append(SystemMessage(content=recovery_message))
                    consecutive_errors = 0  # Reset error count after recovery attempt
                else:
                    continue  # Try again without recovery

        logger.warning(f"Max iterations ({max_iterations}) reached without final response")
        self._log_metrics()
        return "Max iterations reached without generating a final response"

    def _error_recovery(self, state: Dict[str, Any], error_count: int) -> str:
        recovery_prompt = f"""
        The agent has encountered {error_count} consecutive errors. Please analyze the current state and provide a recovery strategy:
        1. Identify potential causes of the errors.
        2. Suggest modifications to the current plan or approach.
        3. Recommend any necessary changes to the agent's configuration or tools.

        Current state:
        {state}

        Provide your recovery strategy in a clear and concise manner.
        """
        
        recovery_message = self.llm.invoke(recovery_prompt)
        logger.info(f"Error recovery strategy: {recovery_message}")
        return f"Error Recovery: {recovery_message}"

    def _log_metrics(self):
        logger.info("Execution Metrics:")
        for key, value in self.metrics.items():
            logger.info(f"{key}: {value}")
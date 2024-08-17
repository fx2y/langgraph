import re
import logging
from typing import List, Dict, Any, Optional, AsyncIterable
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import Counter, Histogram
import asyncio
from functools import lru_cache
import aiohttp
from cachetools import TTLCache
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure tracing and metrics
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Define metrics
plan_counter = meter.create_counter("rewoo_plans_total", "Total number of plans generated")
execution_time = meter.create_histogram("rewoo_execution_time_seconds", "Time taken to execute a plan")
tool_execution_counter = meter.create_counter("rewoo_tool_executions_total", "Total number of tool executions")
tool_execution_time = meter.create_histogram("rewoo_tool_execution_time_seconds", "Time taken to execute a tool")

class ReWOOStep(BaseModel):
    """Represents a single step in the ReWOO plan."""
    plan: str = Field(description="The plan description for this step")
    variable: str = Field(description="The variable name for this step's result")
    tool: str = Field(description="The name of the tool to be used")
    input: str = Field(description="The input for the tool")

class ReWOOState(BaseModel):
    """Represents the state of the ReWOO agent."""
    task: str = Field(description="The original task to be solved")
    plan_string: str = Field(description="The full plan string")
    steps: List[ReWOOStep] = Field(default_factory=list, description="List of steps in the plan")
    results: Dict[str, Any] = Field(default_factory=dict, description="Results of executed steps")
    result: Optional[str] = Field(default=None, description="Final result of the task")

class ReWOOConfig(BaseModel):
    """Configuration for ReWOOAgent."""
    cache_size: int = Field(100, description="Size of the LRU cache for plan results")
    max_retries: int = Field(3, description="Maximum number of retries for failed tool executions")
    timeout: float = Field(30.0, description="Timeout for tool executions in seconds")
    planner_prompt: str = Field(..., description="Prompt template for the planner")
    solver_prompt: str = Field(..., description="Prompt template for the solver")
    plan_cache_ttl: int = Field(3600, description="Time-to-live for cached plans in seconds")

class ReWOOPlugin(ABC):
    @abstractmethod
    async def execute(self, input: Any) -> Any:
        pass

class ReWOOAgent:
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool], config: ReWOOConfig):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.config = config
        self.planner_prompt = ChatPromptTemplate.from_template(config.planner_prompt)
        self.solver_prompt = ChatPromptTemplate.from_template(config.solver_prompt)
        self.session = None
        self.plan_cache = TTLCache(maxsize=config.cache_size, ttl=config.plan_cache_ttl)
        logger.info("ReWOOAgent initialized with %d tools", len(tools))

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @tracer.start_as_current_span("plan")
    @lru_cache(maxsize=100)
    async def plan(self, task: str) -> ReWOOState:
        """Generate a plan for the given task."""
        logger.info("Generating plan for task: %s", task)
        plan_counter.add(1)
        
        cache_key = hash(task)
        if cache_key in self.plan_cache:
            logger.info("Using cached plan for task: %s", task)
            return self.plan_cache[cache_key]

        with execution_time.time():
            tool_descriptions = "\n".join(f"({i+1}) {tool.name}[input]: {tool.description}" 
                                          for i, tool in enumerate(self.tools.values()))
            result = await self.llm.ainvoke(self.planner_prompt.format(task=task, tool_descriptions=tool_descriptions))
            plan_string = result.content
            steps = self._parse_plan(plan_string)
            state = ReWOOState(task=task, plan_string=plan_string, steps=steps)
            
            self.plan_cache[cache_key] = state
            logger.info("Plan generated with %d steps", len(steps))
            return state

    @tracer.start_as_current_span("execute")
    async def execute(self, state: ReWOOState) -> AsyncIterable[ReWOOState]:
        """Execute the plan steps."""
        tasks = [self._execute_step(step, state.results) for step in state.steps]
        for completed_task in asyncio.as_completed(tasks):
            step_result = await completed_task
            state.results.update(step_result)
            yield state

    async def _execute_step(self, step: ReWOOStep, results: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with tracer.start_as_current_span(f"execute_step_{step.variable}"):
                tool_input = self._substitute_variables(step.input, results)
                result = await self._execute_tool_with_retry(step.tool, tool_input)
                logger.info("Executed step %s with result: %s", step.variable, result)
                return {step.variable: str(result)}
        except Exception as e:
            logger.error("Error executing step %s: %s", step.variable, str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise

    @tracer.start_as_current_span("solve")
    async def solve(self, state: ReWOOState) -> ReWOOState:
        """Generate the final solution based on the executed plan."""
        logger.info("Generating final solution for task: %s", state.task)
        plan = "\n".join(f"Plan: {step.plan}\n{step.variable} = {step.tool}[{self._substitute_variables(step.input, state.results)}]"
                         for step in state.steps)
        result = await self.llm.ainvoke(self.solver_prompt.format(plan=plan, task=state.task))
        state.result = result.content
        logger.info("Final solution generated: %s", state.result)
        return state

    @tracer.start_as_current_span("run")
    async def run(self, task: str) -> AsyncIterable[ReWOOState]:
        """Run the full ReWOO workflow for a given task."""
        logger.info("Starting ReWOO workflow for task: %s", task)
        state = await self.plan(task)
        yield state
        async for state in self.execute(state):
            yield state
        state = await self.solve(state)
        yield state
        logger.info("ReWOO workflow completed for task: %s", task)

    def _parse_plan(self, plan_string: str) -> List[ReWOOStep]:
        """Parse the plan string into a list of ReWOOStep objects."""
        regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        matches = re.findall(regex_pattern, plan_string)
        return [ReWOOStep(plan=plan, variable=var, tool=tool, input=input) for plan, var, tool, input in matches]

    def _substitute_variables(self, input_str: str, results: Dict[str, Any]) -> str:
        """Substitute variables in the input string with their corresponding results."""
        for var, value in results.items():
            input_str = input_str.replace(var, str(value))
        return input_str

    async def _execute_tool_with_retry(self, tool_name: str, tool_input: Any) -> Any:
        """Execute a tool with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                with tool_execution_time.time():
                    tool_execution_counter.add(1)
                    return await asyncio.wait_for(self.tools[tool_name].ainvoke(tool_input), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                logger.warning("Tool execution timed out: %s", tool_name)
                if attempt == self.config.max_retries - 1:
                    raise
            except Exception as e:
                logger.error("Tool execution failed: %s. Attempt %d of %d", str(e), attempt + 1, self.config.max_retries)
                if attempt == self.config.max_retries - 1:
                    raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def register_plugin(self, plugin: ReWOOPlugin):
        """Register a new tool plugin."""
        self.tools[plugin.name] = plugin
        logger.info("Registered new plugin: %s", plugin.name)

    async def collect_feedback(self, task: str, state: ReWOOState, feedback: str) -> None:
        """Collect feedback for continuous improvement."""
        logger.info(f"Received feedback for task '{task}': {feedback}")
        # Implement logic to adjust tool selection or planning based on feedback
        # This could involve updating a feedback database or adjusting tool weights

# Example usage
async def main():
    from langchain_openai import ChatOpenAI
    from langchain_community.tools import DuckDuckGoSearchRun

    config = ReWOOConfig(
        planner_prompt="Your planner prompt here",
        solver_prompt="Your solver prompt here",
    )
    
    llm = ChatOpenAI()
    tools = [DuckDuckGoSearchRun()]
    
    async with ReWOOAgent(llm, tools, config) as agent:
        async for state in agent.run("What is the capital of France?"):
            print(f"Current state: {state.model_dump_json()}")

if __name__ == "__main__":
    asyncio.run(main())
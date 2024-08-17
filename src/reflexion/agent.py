import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field, ValidationError
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import PydanticToolsParser
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure tracing and metrics
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Define metrics
answer_generation_counter = meter.create_counter("answer_generation_count")
answer_generation_time = meter.create_histogram("answer_generation_time")
revision_counter = meter.create_counter("revision_count")
revision_time = meter.create_histogram("revision_time")

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )

class ReviseAnswer(AnswerQuestion):
    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )

class ReflexionAgentConfig(BaseModel):
    max_iterations: int = Field(5, ge=1)
    max_retries: int = Field(3, ge=1)
    retry_delay: float = Field(1.0, ge=0)

class ReflexionAgent:
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        config: ReflexionAgentConfig = ReflexionAgentConfig()
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.config = config
        self._setup_chains()
        logger.info("ReflexionAgent initialized with %d tools", len(tools))

    def _setup_chains(self):
        actor_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are expert researcher. Current time: {time}\n\n"
                       "1. {first_instruction}\n"
                       "2. Reflect and critique your answer. Be severe to maximize improvement.\n"
                       "3. Recommend search queries to research information and improve your answer."),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "\n\n<system>Reflect on the user's original question and the"
                     " actions taken thus far. Respond using the {function_name} function.</reminder>"),
        ])

        self.initial_answer_chain = actor_prompt_template.partial(
            first_instruction="Provide a detailed ~250 word answer.",
            function_name=AnswerQuestion.__name__,
        ) | self.llm.bind_tools(tools=[AnswerQuestion])

        self.revision_chain = actor_prompt_template.partial(
            first_instruction="Revise your previous answer using the new information.",
            function_name=ReviseAnswer.__name__,
        ) | self.llm.bind_tools(tools=[ReviseAnswer])

        self.initial_validator = PydanticToolsParser(tools=[AnswerQuestion])
        self.revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @tracer.start_as_current_span("generate_answer")
    async def generate_answer(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Generating initial answer")
        start_time = asyncio.get_event_loop().time()
        try:
            response = await self.initial_answer_chain.ainvoke({"messages": messages})
            self.initial_validator.invoke(response)
            return response
        except ValidationError as e:
            logger.error("Validation error in generate_answer: %s", str(e))
            raise
        except Exception as e:
            logger.error("Error in generate_answer: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            end_time = asyncio.get_event_loop().time()
            answer_generation_counter.add(1)
            answer_generation_time.record(end_time - start_time)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @tracer.start_as_current_span("revise_answer")
    async def revise_answer(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Revising answer")
        start_time = asyncio.get_event_loop().time()
        try:
            response = await self.revision_chain.ainvoke({"messages": messages})
            self.revision_validator.invoke(response)
            return response
        except ValidationError as e:
            logger.error("Validation error in revise_answer: %s", str(e))
            raise
        except Exception as e:
            logger.error("Error in revise_answer: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            end_time = asyncio.get_event_loop().time()
            revision_counter.add(1)
            revision_time.record(end_time - start_time)

    @tracer.start_as_current_span("execute_tool")
    async def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        logger.info("Executing tool: %s", tool_name)
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        try:
            result = await self.tools[tool_name].arun(**kwargs)
            logger.info("Tool execution completed: %s", tool_name)
            return result
        except Exception as e:
            logger.error("Error executing tool '%s': %s", tool_name, str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise

    @tracer.start_as_current_span("run")
    async def run(self, query: str) -> Dict[str, Any]:
        logger.info("Starting ReflexionAgent workflow for query: %s", query)
        messages = [HumanMessage(content=query)]
        try:
            for iteration in range(self.config.max_iterations):
                if iteration == 0:
                    response = await self.generate_answer(messages)
                else:
                    response = await self.revise_answer(messages)

                tool_name = response.tool_calls[0]["name"]
                tool_args = json.loads(response.tool_calls[0]["arguments"])
                tool_result = await self.execute_tool(tool_name, **tool_args)

                messages.extend([
                    response,
                    ToolMessage(
                        content=json.dumps(tool_result),
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ])

                if iteration == self.config.max_iterations - 1:
                    break

            final_answer = json.loads(response.tool_calls[0]["arguments"])
            logger.info("ReflexionAgent workflow completed successfully")
            return final_answer
        except Exception as e:
            logger.error("Error in ReflexionAgent workflow: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise

# Example usage
async def main():
    from langchain_anthropic import ChatAnthropic
    from langchain_community.tools.tavily_search import TavilySearchResults

    llm = ChatAnthropic(model="claude-3-sonnet-20240229")
    search_tool = TavilySearchResults(max_results=5)
    
    agent = ReflexionAgent(
        llm=llm,
        tools=[
            StructuredTool.from_function(
                search_tool.invoke,
                name=AnswerQuestion.__name__,
                description="Run search queries to find information"
            ),
            StructuredTool.from_function(
                search_tool.invoke,
                name=ReviseAnswer.__name__,
                description="Run search queries to find information for revision"
            ),
        ],
        config=ReflexionAgentConfig(max_iterations=3)
    )

    result = await agent.run("How should we handle the climate crisis?")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import Counter, Histogram
from functools import lru_cache
from abc import ABC, abstractmethod
import aioredis
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure tracing and metrics
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Define metrics
document_retrieval_counter = meter.create_counter("document_retrieval_count")
document_retrieval_time = meter.create_histogram("document_retrieval_time")
grading_counter = meter.create_counter("grading_count")
grading_time = meter.create_histogram("grading_time")
answer_generation_counter = meter.create_counter("answer_generation_count")
answer_generation_time = meter.create_histogram("answer_generation_time")

class DocumentGrade(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    is_relevant: bool = Field(description="Whether the documents are relevant to the question")
    reasoning: str = Field(description="Explanation for the relevance assessment")

class ReActAgentConfig(BaseModel):
    """Configuration for ReActAgent."""
    max_documents: int = 5
    relevance_threshold: float = 0.7
    cache_size: int = 100
    redis_url: Optional[str] = None  # Add this field for Redis URL
    max_retries: int = 3
    retry_delay: float = 1.0

class ToolPlugin(ABC):
    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        pass

# Assuming you have a custom tool plugin implementation
class CustomToolPlugin(ToolPlugin):
    async def execute(self, **kwargs: Any) -> Any:
        # Custom execution logic
        pass

# Assuming you have a custom retriever implementation
class CustomRetriever(BaseRetriever):
    async def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        # Custom logic to fetch relevant documents based on the query
        pass

# Assuming you have a custom language model implementation
class CustomLanguageModel(BaseLanguageModel):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        # Custom logic to generate text based on the prompt
        pass

# Initialize the language model
llm = CustomLanguageModel()

class ReActAgent:
    def __init__(
        self,
        llm: BaseLanguageModel,  # Ensure this is of type BaseLanguageModel
        retriever: BaseRetriever,  # Ensure this is of type BaseRetriever
        tools: List[BaseTool],
        grading_prompt: ChatPromptTemplate,
        config: ReActAgentConfig = ReActAgentConfig()
    ):
        self.llm = llm  # Use the provided language model
        self.retriever = retriever  # Use the provided retriever
        self.tools = tools
        self.grading_prompt = grading_prompt
        self.config = config
        self.plugins: Dict[str, ToolPlugin] = {}
        self.redis: Optional[aioredis.Redis] = None
        if self.config.redis_url:
            self.redis = aioredis.from_url(self.config.redis_url)  # Initialize Redis client if URL is provided
        logger.info("ReActAgent initialized with %d tools", len(tools))

    def register_plugin(self, name: str, plugin: ToolPlugin):
        """Register a new tool plugin."""
        self.plugins[name] = plugin
        logger.info(f"Registered new plugin: {name}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @tracer.start_as_current_span("retrieve_documents")
    async def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve documents based on the query."""
        logger.info("Retrieving documents for query: %s", query)
        start_time = asyncio.get_event_loop().time()
        try:
            if self.redis:
                cached_result = await self.redis.get(f"document_cache:{query}")
                if cached_result:
                    return eval(cached_result)

            documents = await self.retriever.get_relevant_documents(query)
            documents = documents[:self.config.max_documents]
            result = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]

            if self.redis:
                await self.redis.set(f"document_cache:{query}", str(result), ex=3600)

            logger.info("Retrieved %d documents", len(documents))
            return result
        except Exception as e:
            logger.error("Error retrieving documents: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            end_time = asyncio.get_event_loop().time()
            document_retrieval_counter.add(1)
            document_retrieval_time.record(end_time - start_time)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @tracer.start_as_current_span("grade_document_retrieval")
    async def grade_document_retrieval(self, question: str, documents: List[Dict[str, Any]]) -> DocumentGrade:
        """Grade the retrieved documents for relevance."""
        logger.info("Grading document retrieval for question: %s", question)
        start_time = asyncio.get_event_loop().time()
        try:
            grader = self.llm.with_structured_output(DocumentGrade)
            result = await grader.invoke(
                self.grading_prompt.format(question=question, documents=documents)
            )
            logger.info("Document grading result: %s", result)
            return result
        except Exception as e:
            logger.error("Error grading documents: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            end_time = asyncio.get_event_loop().time()
            grading_counter.add(1)
            grading_time.record(end_time - start_time)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @tracer.start_as_current_span("execute_tool")
    async def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a specific tool."""
        logger.info("Executing tool: %s", tool_name)
        if tool_name in self.plugins:
            try:
                result = await self.plugins[tool_name].execute(**kwargs)
                logger.info("Plugin execution completed: %s", tool_name)
                return result
            except Exception as e:
                logger.error("Error executing plugin '%s': %s", tool_name, str(e))
                trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
                raise
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            logger.error("Tool '%s' not found", tool_name)
            raise ValueError(f"Tool '{tool_name}' not found")
        try:
            result = await tool.run(**kwargs)
            logger.info("Tool execution completed: %s", tool_name)
            return result
        except Exception as e:
            logger.error("Error executing tool '%s': %s", tool_name, str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @tracer.start_as_current_span("generate_answer")
    async def generate_answer(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """Generate an answer based on the question and retrieved documents."""
        logger.info("Generating answer for question: %s", question)
        start_time = asyncio.get_event_loop().time()
        prompt = ChatPromptTemplate.from_template(
            "Answer the following question using the provided documents: {question}\n\nDocuments: {documents}"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            answer = await chain.invoke({"question": question, "documents": documents})
            logger.info("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error("Error generating answer: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            end_time = asyncio.get_event_loop().time()
            answer_generation_counter.add(1)
            answer_generation_time.record(end_time - start_time)

    @tracer.start_as_current_span("run")
    async def run(self, query: str) -> Dict[str, Any]:
        """Execute the ReAct agent's workflow."""
        logger.info("Starting ReAct agent workflow for query: %s", query)
        try:
            documents = await self.retrieve_documents(query)
            grade = await self.grade_document_retrieval(query, documents)
            
            if grade.is_relevant:
                answer = await self.generate_answer(query, documents)
            else:
                logger.warning("No relevant documents found for query: %s", query)
                answer = "I couldn't find relevant information to answer your question."

            result = {
                "query": query,
                "documents": documents,
                "grade": grade.dict(),
                "answer": answer
            }
            logger.info("ReAct agent workflow completed successfully")
            return result
        except Exception as e:
            logger.error("Error in ReAct agent workflow: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise

    async def close(self):
        """Close any open connections."""
        if self.redis:
            await self.redis.close()

# Register the plugin with the agent
agent = ReActAgent(llm, retriever, tools, grading_prompt)
agent.register_plugin("custom_tool", CustomToolPlugin())
import logging
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field, validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import aiohttp
import asyncio
from functools import lru_cache
import time

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
web_search_counter = meter.create_counter("web_search_count")
web_search_time = meter.create_histogram("web_search_time")

class DocumentGrade(BaseModel):
    """Score for relevance check on retrieved documents."""
    relevance_score: float = Field(description="Relevance score of the documents (0.0 to 1.0)")
    reasoning: str = Field(description="Explanation for the relevance assessment")

    @validator('relevance_score')
    def check_relevance_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('relevance_score must be between 0 and 1')
        return v

class CustomLangGraphAgentConfig(BaseModel):
    """Configuration for CustomLangGraphAgent."""
    max_documents: int = Field(5, ge=1)
    relevance_threshold: float = Field(0.7, ge=0, le=1)
    max_retries: int = Field(3, ge=1)
    retry_delay: float = Field(1.0, ge=0)
    cache_size: int = Field(100, ge=1)
    timeout: float = Field(30.0, ge=0)
    rate_limit: int = Field(10, ge=1)  # requests per second

class CustomTool(BaseModel):
    name: str
    description: str
    func: Callable

class CustomLangGraphAgent:
    def __init__(
        self,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        grading_prompt: ChatPromptTemplate,
        config: CustomLangGraphAgentConfig = CustomLangGraphAgentConfig()
    ):
        self.llm = llm
        self.retriever = retriever
        self.grading_prompt = grading_prompt
        self.config = config
        self.session = None
        self.rate_limiter = asyncio.Semaphore(self.config.rate_limit)
        self.web_search_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.custom_tools: Dict[str, CustomTool] = {}
        logger.info("CustomLangGraphAgent initialized")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError)
    )
    @tracer.start_as_current_span("retrieve")
    @lru_cache(maxsize=100)
    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve documents based on the query."""
        logger.info("Retrieving documents for query: %s", query)
        start_time = time.time()
        try:
            async with self.rate_limiter:
                documents = await asyncio.wait_for(
                    self.retriever.aget_relevant_documents(query),
                    timeout=self.config.timeout
                )
            documents = documents[:self.config.max_documents]
            result = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
            logger.info("Retrieved %d documents", len(documents))
            return result
        except asyncio.TimeoutError:
            logger.error("Timeout while retrieving documents")
            raise
        except Exception as e:
            logger.error("Error retrieving documents: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            end_time = time.time()
            document_retrieval_counter.add(1)
            document_retrieval_time.record(end_time - start_time)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError)
    )
    @tracer.start_as_current_span("grade_documents")
    async def grade_documents(self, question: str, documents: List[Dict[str, Any]]) -> DocumentGrade:
        """Grade the retrieved documents for relevance."""
        logger.info("Grading document retrieval for question: %s", question)
        start_time = time.time()
        try:
            async with self.rate_limiter:
                grader = self.llm.with_structured_output(DocumentGrade)
                result = await asyncio.wait_for(
                    grader.ainvoke(
                        self.grading_prompt.format(question=question, documents=documents)
                    ),
                    timeout=self.config.timeout
                )
            logger.info("Document grading result: %s", result)
            return result
        except asyncio.TimeoutError:
            logger.error("Timeout while grading documents")
            raise
        except Exception as e:
            logger.error("Error grading documents: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            end_time = time.time()
            grading_counter.add(1)
            grading_time.record(end_time - start_time)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError)
    )
    @tracer.start_as_current_span("generate")
    async def generate(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """Generate an answer based on the question and retrieved documents."""
        logger.info("Generating answer for question: %s", question)
        start_time = time.time()
        prompt = ChatPromptTemplate.from_template(
            "Answer the following question using the provided documents: {question}\n\nDocuments: {documents}"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            async with self.rate_limiter:
                answer = await asyncio.wait_for(
                    chain.ainvoke({"question": question, "documents": documents}),
                    timeout=self.config.timeout
                )
            logger.info("Answer generated successfully")
            return answer
        except asyncio.TimeoutError:
            logger.error("Timeout while generating answer")
            raise
        except Exception as e:
            logger.error("Error generating answer: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            end_time = time.time()
            answer_generation_counter.add(1)
            answer_generation_time.record(end_time - start_time)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError)
    )
    @tracer.start_as_current_span("web_search")
    async def web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a web search for additional information."""
        logger.info("Performing web search for query: %s", query)
        start_time = time.time()
        
        if query in self.web_search_cache:
            logger.info("Returning cached web search results")
            return self.web_search_cache[query]
        
        try:
            async with self.rate_limiter:
                async with self.session.get(
                    "https://api.search.com",
                    params={"q": query},
                    timeout=self.config.timeout
                ) as response:
                    response.raise_for_status()
                    search_results = await response.json()
            
            results = [
                {"content": result["snippet"], "metadata": {"url": result["url"]}}
                for result in search_results["results"][:self.config.max_documents]
            ]
            logger.info("Web search completed with %d results", len(results))
            
            self.web_search_cache[query] = results
            return results
        except asyncio.TimeoutError:
            logger.error("Timeout while performing web search")
            raise
        except Exception as e:
            logger.error("Error performing web search: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            end_time = time.time()
            web_search_counter.add(1)
            web_search_time.record(end_time - start_time)

    def update_config(self, **kwargs):
        """Update the agent's configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
        self.rate_limiter = asyncio.Semaphore(self.config.rate_limit)
        logger.info("Configuration updated")

    def clear_cache(self):
        """Clear the web search cache."""
        self.web_search_cache.clear()
        logger.info("Web search cache cleared")

    def register_custom_tool(self, tool: CustomTool):
        """Register a custom tool."""
        self.custom_tools[tool.name] = tool
        logger.info(f"Custom tool registered: {tool.name}")

    async def execute_custom_tool(self, tool_name: str, **kwargs):
        """Execute a custom tool."""
        if tool_name not in self.custom_tools:
            raise ValueError(f"Custom tool not found: {tool_name}")
        
        tool = self.custom_tools[tool_name]
        logger.info(f"Executing custom tool: {tool_name}")
        
        try:
            result = await tool.func(**kwargs)
            logger.info(f"Custom tool execution completed: {tool_name}")
            return result
        except Exception as e:
            logger.error(f"Error executing custom tool '{tool_name}': {str(e)}")
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise

    @tracer.start_as_current_span("run")
    async def run(self, query: str) -> Dict[str, Any]:
        """Execute the CustomLangGraphAgent's workflow."""
        logger.info("Starting CustomLangGraphAgent workflow for query: %s", query)
        try:
            documents = await self.retrieve(query)
            grade = await self.grade_documents(query, documents)
            
            if grade.relevance_score >= self.config.relevance_threshold:
                answer = await self.generate(query, documents)
            else:
                logger.warning("No relevant documents found. Performing web search.")
                web_results = await self.web_search(query)
                answer = await self.generate(query, web_results)

            result = {
                "query": query,
                "documents": documents,
                "grade": grade.dict(),
                "answer": answer
            }
            logger.info("CustomLangGraphAgent workflow completed successfully")
            return result
        except Exception as e:
            logger.error("Error in CustomLangGraphAgent workflow: %s", str(e))
            trace.get_current_span().set_status(Status(StatusCode.ERROR, str(e)))
            raise

    async def close(self):
        """Close any open connections."""
        if self.session:
            await self.session.close()
        logger.info("CustomLangGraphAgent resources cleaned up")
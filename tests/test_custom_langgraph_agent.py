import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp
import asyncio
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure tracing
tracer = trace.get_tracer(__name__)

class DocumentGrade(BaseModel):
    """Score for relevance check on retrieved documents."""
    relevance_score: float = Field(description="Relevance score of the documents (0.0 to 1.0)")
    reasoning: str = Field(description="Explanation for the relevance assessment")

class CustomLangGraphAgentConfig(BaseModel):
    """Configuration for CustomLangGraphAgent."""
    max_documents: int = 5
    relevance_threshold: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_size: int = 100
    timeout: float = 30.0
    rate_limit: int = 10

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
        logger.info("CustomLangGraphAgent initialized")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    @tracer.start_as_current_span("web_search")
    async def web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a web search for additional information."""
        logger.info("Performing web search for query: %s", query)
        start_time = time.time()
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

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.agents.custom_langgraph_agent import CustomLangGraphAgent, CustomLangGraphAgentConfig, DocumentGrade

@pytest.fixture
def agent_config():
    return CustomLangGraphAgentConfig()

@pytest.fixture
def mock_llm():
    return AsyncMock()

@pytest.fixture
def mock_retriever():
    return AsyncMock()

@pytest.fixture
def mock_grading_prompt():
    return AsyncMock()

@pytest.fixture
async def agent(agent_config, mock_llm, mock_retriever, mock_grading_prompt):
    async with CustomLangGraphAgent(mock_llm, mock_retriever, mock_grading_prompt, agent_config) as agent:
        yield agent

@pytest.mark.asyncio
async def test_retrieve(agent, mock_retriever):
    mock_retriever.aget_relevant_documents.return_value = [
        AsyncMock(page_content="content1", metadata={"source": "source1"}),
        AsyncMock(page_content="content2", metadata={"source": "source2"}),
    ]
    
    result = await agent.retrieve("test query")
    
    assert len(result) == 2
    assert result[0]["content"] == "content1"
    assert result[0]["metadata"] == {"source": "source1"}
    assert result[1]["content"] == "content2"
    assert result[1]["metadata"] == {"source": "source2"}

@pytest.mark.asyncio
async def test_grade_documents(agent, mock_llm):
    mock_llm.with_structured_output.return_value.ainvoke.return_value = DocumentGrade(
        relevance_score=0.8,
        reasoning="The documents are highly relevant."
    )
    
    result = await agent.grade_documents("test question", [{"content": "test content"}])
    
    assert result.relevance_score == 0.8
    assert result.reasoning == "The documents are highly relevant."

@pytest.mark.asyncio
async def test_generate(agent, mock_llm):
    mock_llm.__or__.return_value.__or__.return_value.ainvoke.return_value = "Generated answer"
    
    result = await agent.generate("test question", [{"content": "test content"}])
    
    assert result == "Generated answer"

@pytest.mark.asyncio
async def test_web_search(agent):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value.__aenter__.return_value.json.return_value = {
            "results": [
                {"snippet": "result1", "url": "url1"},
                {"snippet": "result2", "url": "url2"},
            ]
        }
        
        result = await agent.web_search("test query")
        
        assert len(result) == 2
        assert result[0]["content"] == "result1"
        assert result[0]["metadata"] == {"url": "url1"}
        assert result[1]["content"] == "result2"
        assert result[1]["metadata"] == {"url": "url2"}

@pytest.mark.asyncio
async def test_run_with_relevant_documents(agent):
    agent.retrieve = AsyncMock(return_value=[{"content": "test content"}])
    agent.grade_documents = AsyncMock(return_value=DocumentGrade(relevance_score=0.8, reasoning="Relevant"))
    agent.generate = AsyncMock(return_value="Generated answer")
    
    result = await agent.run("test query")
    
    assert result["query"] == "test query"
    assert result["documents"] == [{"content": "test content"}]
    assert result["grade"] == {"relevance_score": 0.8, "reasoning": "Relevant"}
    assert result["answer"] == "Generated answer"
    agent.web_search.assert_not_called()

@pytest.mark.asyncio
async def test_run_with_web_search(agent):
    agent.retrieve = AsyncMock(return_value=[{"content": "test content"}])
    agent.grade_documents = AsyncMock(return_value=DocumentGrade(relevance_score=0.5, reasoning="Not relevant"))
    agent.web_search = AsyncMock(return_value=[{"content": "web content"}])
    agent.generate = AsyncMock(return_value="Generated answer from web")
    
    result = await agent.run("test query")
    
    assert result["query"] == "test query"
    assert result["documents"] == [{"content": "test content"}]
    assert result["grade"] == {"relevance_score": 0.5, "reasoning": "Not relevant"}
    assert result["answer"] == "Generated answer from web"
    agent.web_search.assert_called_once_with("test query")

@pytest.mark.asyncio
async def test_error_handling(agent):
    agent.retrieve = AsyncMock(side_effect=Exception("Test error"))
    
    with pytest.raises(Exception):
        await agent.run("test query")

@pytest.mark.asyncio
async def test_timeout_handling(agent):
    agent.retrieve = AsyncMock(side_effect=asyncio.TimeoutError())
    
    with pytest.raises(asyncio.TimeoutError):
        await agent.run("test query")
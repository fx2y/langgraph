import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.rewoo.agent import ReWOOAgent, ReWOOConfig, ReWOOState, ReWOOStep

@pytest.fixture
def config():
    return ReWOOConfig(
        planner_prompt="Test planner prompt",
        solver_prompt="Test solver prompt",
    )

@pytest.fixture
def mock_llm():
    return AsyncMock()

@pytest.fixture
def mock_tool():
    tool = AsyncMock()
    tool.name = "TestTool"
    tool.description = "A test tool"
    return tool

@pytest.fixture
async def agent(config, mock_llm, mock_tool):
    async with ReWOOAgent(mock_llm, [mock_tool], config) as agent:
        yield agent

@pytest.mark.asyncio
async def test_plan(agent, mock_llm):
    mock_llm.ainvoke.return_value.content = "Plan: Test plan\n#E1 = TestTool[test input]"
    
    result = await agent.plan("Test task")
    
    assert isinstance(result, ReWOOState)
    assert len(result.steps) == 1
    assert result.steps[0].tool == "TestTool"
    assert result.steps[0].input == "test input"

@pytest.mark.asyncio
async def test_execute(agent, mock_tool):
    state = ReWOOState(
        task="Test task",
        plan_string="Test plan",
        steps=[ReWOOStep(plan="Test step", variable="#E1", tool="TestTool", input="test input")]
    )
    mock_tool.ainvoke.return_value = "Test result"
    
    results = [s async for s in agent.execute(state)]
    
    assert len(results) == 1
    assert results[0].results["#E1"] == "Test result"

@pytest.mark.asyncio
async def test_solve(agent, mock_llm):
    state = ReWOOState(
        task="Test task",
        plan_string="Test plan",
        steps=[ReWOOStep(plan="Test step", variable="#E1", tool="TestTool", input="test input")],
        results={"#E1": "Test result"}
    )
    mock_llm.ainvoke.return_value.content = "Final answer"
    
    result = await agent.solve(state)
    
    assert result.result == "Final answer"

@pytest.mark.asyncio
async def test_run(agent, mock_llm, mock_tool):
    mock_llm.ainvoke.side_effect = [
        AsyncMock(content="Plan: Test plan\n#E1 = TestTool[test input]"),
        AsyncMock(content="Final answer")
    ]
    mock_tool.ainvoke.return_value = "Test result"
    
    results = [s async for s in agent.run("Test task")]
    
    assert len(results) == 3
    assert results[0].plan_string == "Plan: Test plan\n#E1 = TestTool[test input]"
    assert results[1].results["#E1"] == "Test result"
    assert results[2].result == "Final answer"

@pytest.mark.asyncio
async def test_error_handling(agent, mock_tool):
    state = ReWOOState(
        task="Test task",
        plan_string="Test plan",
        steps=[ReWOOStep(plan="Test step", variable="#E1", tool="TestTool", input="test input")]
    )
    mock_tool.ainvoke.side_effect = [Exception("Test error"), "Test result"]
    
    results = [s async for s in agent.execute(state)]
    
    assert len(results) == 1
    assert results[0].results["#E1"] == "Test result"

@pytest.mark.asyncio
async def test_register_plugin(agent):
    new_tool = AsyncMock()
    new_tool.name = "NewTool"
    new_tool.description = "A new test tool"
    
    agent.register_plugin(new_tool)
    
    assert "NewTool" in agent.tools
    assert agent.tools["NewTool"] == new_tool
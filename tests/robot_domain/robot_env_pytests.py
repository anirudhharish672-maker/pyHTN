import pytest
from copy import deepcopy

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from py_htn.domain import Task, Method, Operator
from py_htn.fact import Fact
from py_htn.conditions import AND, OR, NOT, Filter
from py_htn.common import V
from py_htn.exceptions import StopException, FailedPlanException
from py_htn.planner import HtnPlanner, PlanNode


# Mock agent for testing
class MockAgent:
    def __init__(self):
        self.env = MockEnvironment()


class MockEnvironment:
    def __init__(self):
        self.state = []
        self.execution_log = []

    def execute_action(self, operator):
        """Record the execution and update state"""
        self.execution_log.append(operator.name)

        # Simple state updates based on operator type
        if operator.name == 'move':
            from_loc = operator.args[0].name if hasattr(operator.args[0], 'name') else operator.args[0]
            to_loc = operator.args[1].name if hasattr(operator.args[1], 'name') else operator.args[1]

            # Update robot location
            for fact in self.state:
                if fact.get('type') == 'robot':
                    fact['location'] = to_loc

        elif operator.name == 'pickup':
            item = operator.args[0].name if hasattr(operator.args[0], 'name') else operator.args[0]

            # Remove item from state
            self.state = [fact for fact in self.state if not (
                    fact.get('type') == 'item' and fact.get('name') == item
            )]

            # Update robot state
            for fact in self.state:
                if fact.get('type') == 'robot':
                    fact['holding'] = item
                    if 'gripper' in fact:
                        del fact['gripper']

        elif operator.name == 'drop':
            item = operator.args[0].name if hasattr(operator.args[0], 'name') else operator.args[0]
            robot_loc = None

            # Find robot location and update holding
            for fact in self.state:
                if fact.get('type') == 'robot':
                    robot_loc = fact.get('location')
                    if 'holding' in fact:
                        del fact['holding']
                    fact['gripper'] = 'empty'

            # Add item to current location
            if robot_loc:
                new_fact = Fact(id=f"i-{item}", type="item", name=item, location=robot_loc)
                self.state.append(new_fact)

    def get_state(self):
        """Return the current state"""
        return self.state


# Test fixtures

@pytest.fixture
def robot_domain():
    """Create a simple robot domain for testing"""
    return {
        "move/2": [
            Operator(
                head=("move", V("from_loc"), V("to_loc")),
                preconditions=AND(
                    Fact(type="location", name=V("from_loc")),
                    Fact(type="location", name=V("to_loc")),
                    Fact(type="robot", location=V("from_loc"))
                ),
                effects=AND(
                    Fact(type="robot", location=V("to_loc")),
                    NOT(Fact(type="robot", location=V("from_loc")))
                )
            )
        ],
        "pickup/1": [
            Operator(
                head=("pickup", V("item")),
                preconditions=AND(
                    Fact(type="item", name=V("item"), location=V("loc")),
                    Fact(type="robot", location=V("loc")),
                    Fact(type="robot", gripper="empty")
                ),
                effects=AND(
                    Fact(type="robot", holding=V("item")),
                    NOT(Fact(type="robot", gripper="empty")),
                    NOT(Fact(type="item", name=V("item"), location=V("loc")))
                )
            )
        ],
        "drop/1": [
            Operator(
                head=("drop", V("item")),
                preconditions=AND(
                    Fact(type="robot", holding=V("item")),
                    Fact(type="robot", location=V("loc"))
                ),
                effects=AND(
                    Fact(type="item", name=V("item"), location=V("loc")),
                    Fact(type="robot", gripper="empty"),
                    NOT(Fact(type="robot", holding=V("item")))
                )
            )
        ],
        "get_item/1": [
            Method(
                head=("get_item", V("item")),
                preconditions=AND(
                    Fact(type="item", name=V("item"), location=V("item_loc")),
                    Fact(type="robot", location=V("robot_loc"))
                ),
                subtasks=[
                    Task(name="move", args=(V("robot_loc"), V("item_loc"))),
                    Task(name="pickup", args=(V("item"),))
                ]
            )
        ],
        "deliver_item/2": [
            Method(
                head=("deliver_item", V("item"), V("dest")),
                preconditions=AND(
                    Fact(type="item", name=V("item")),
                    Fact(type="location", name=V("dest"))
                ),
                subtasks=[
                    Task(name="get_item", args=(V("item"),)),
                    Task(name="move", args=(V("curr_loc"), V("dest"))),
                    Task(name="drop", args=(V("item"),))
                ]
            )
        ]
    }


@pytest.fixture
def initial_state():
    """Create an initial state for testing"""
    return [
        Fact(id="l1", type="location", name="kitchen"),
        Fact(id="l2", type="location", name="living_room"),
        Fact(id="l3", type="location", name="bedroom"),
        Fact(id="r1", type="robot", location="kitchen", gripper="empty"),
        Fact(id="i1", type="item", name="book", location="living_room")
    ]


@pytest.fixture
def deliver_tasks():
    """Create delivery tasks for testing"""
    return [
        {"task": "deliver_item", "arguments": ("book", "bedroom"), "priority": "high"}
    ]


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing"""
    return MockAgent()


@pytest.fixture
def planner(robot_domain, deliver_tasks, mock_agent, initial_state):
    """Create a NodePlanner instance for testing"""
    planner = HtnPlanner(tasks=deliver_tasks, domain=robot_domain, agent=mock_agent, logging=True)
    mock_agent.env.state = deepcopy(initial_state)
    planner.update_state(initial_state)
    return planner


# Tests
def test_planner_initialization(planner, deliver_tasks):
    """Test that the planner initializes correctly"""
    assert len(planner.root_tasks) == 1
    assert planner.root_tasks[0].content.name == "deliver_item"
    assert len(planner.root_tasks[0].content.args) == 2
    assert planner.root_tasks[0].content.args[0] == "book"
    assert planner.root_tasks[0].content.args[1] == "bedroom"


def test_get_next_decomposition(planner):
    """Test that get_next_decomposition returns valid options"""
    decomposition = planner.get_next_decomposition()

    assert 'task_node' in decomposition
    assert 'options' in decomposition
    assert len(decomposition['options']) > 0
    assert decomposition['task_node'].content.name == "deliver_item"


def test_apply_method(planner):
    """Test applying a method to a task"""
    decomposition = planner.get_next_decomposition()
    task_node = decomposition['task_node']

    # Apply the deliver_item method
    result = planner.apply(task_node.id, 0)

    assert result['node_type'] == 'method'
    assert result['name'] == 'deliver_item'
    assert len(result['subtask_ids']) == 3  # Should create 3 subtasks

    # Check that tracking info is updated
    assert planner.planning_depth > 0
    assert len(planner.current_task_path) > 1
    assert planner.last_decomposition['type'] == 'method'
    assert planner.last_decomposition['applied'] == 'deliver_item'


def test_apply_operator(planner):
    """Test applying an operator"""
    # First set up the state to have a valid operator
    decomposition = planner.get_next_decomposition()
    task_node = decomposition['task_node']

    # Apply the deliver_item method
    method_result = planner.apply(task_node.id, 0)

    # Apply the get_item method
    next_task_id = method_result['next_task_id']
    method_result = planner.apply(next_task_id, 0)

    # Now we should have a move operator as the next task
    next_task_id = method_result['next_task_id']

    # Get decomposition for the move task
    decomposition = planner.get_next_decomposition(next_task_id)

    # Apply the move operator
    result = planner.apply(next_task_id, 0)

    assert result['node_type'] == 'operator'
    assert result['name'] == 'move'

    # Check that the agent executed the action
    assert planner.agent.env.execution_log[-1] == 'move'

    # Check that tracking info is updated
    assert 'move' in planner.get_current_operator_trace()
    assert len(planner.execution_history) > 0


def test_complete_plan(planner, mock_agent, initial_state):
    """Test the complete planning and execution process"""
    # Reset the state
    mock_agent.env.state = deepcopy(initial_state)
    mock_agent.env.execution_log = []
    planner.update_state(initial_state)

    # Run the full plan
    plan = planner.plan(initial_state)

    # Check that operators were executed in the plan
    assert len(plan) > 0
    print(plan)

    # Check that all actions were executed
    expected_actions = ['move', 'pickup', 'move', 'drop']
    assert mock_agent.env.execution_log == expected_actions

    # Check the final state
    robot_fact = next((f for f in mock_agent.env.state if f.get('type') == 'robot'), None)
    book_fact = next((f for f in mock_agent.env.state if f.get('type') == 'item' and f.get('name') == 'book'), None)

    assert robot_fact['location'] == 'bedroom'
    assert robot_fact['gripper'] == 'empty'
    assert book_fact['location'] == 'bedroom'


def test_tracking_info(planner, initial_state):
    """Test that tracking information is maintained correctly"""
    # Reset state
    planner.update_state(initial_state)

    # Step through the plan
    decomposition = planner.get_next_decomposition()
    task_node = decomposition['task_node']

    # Apply the deliver_item method
    result = planner.apply(task_node.id, 0)

    # Check tracking info after method application
    assert len(planner.current_task_path) > 0
    assert planner.planning_depth > 0
    assert len(planner.execution_history) == 1
    assert planner.execution_history[0]['type'] == 'method'

    # Apply the get_item method
    next_task_id = result['next_task_id']
    result = planner.apply(next_task_id, 0)

    # Check tracking info after second method
    assert len(planner.execution_history) == 2
    assert planner.get_current_method_trace() != "No methods applied yet"

    # Apply move operator
    next_task_id = result['next_task_id']
    result = planner.apply(next_task_id, 0)

    # Check tracking after operator
    assert len(planner.execution_history) == 3
    assert planner.get_current_operator_trace() != "No operators applied yet"

    # Check task hierarchy
    hierarchy = planner.get_task_hierarchy_string()
    assert "deliver_item" in hierarchy
    assert "get_item" in hierarchy
    assert "pickup" in hierarchy  # Should be the current task


def test_error_handling(planner):
    """Test error handling for invalid inputs"""
    # Test with non-existent task ID
    with pytest.raises(ValueError):
        planner.apply("nonexistent-id", 0)

    # Test with invalid method/operator index
    decomposition = planner.get_next_decomposition()
    task_node = decomposition['task_node']

    with pytest.raises(ValueError):
        planner.apply(task_node.id, 999)  # Invalid index


def test_visualization(planner, initial_state):
    """Test the visualization functionality"""
    # Reset state
    planner.update_state(initial_state)

    # Run a partial plan
    decomposition = planner.get_next_decomposition()
    task_node = decomposition['task_node']

    # Apply the deliver_item method
    result = planner.apply(task_node.id, 0)

    # Get visualization
    viz = planner.visualize()

    # Basic checks on visualization output
    assert viz
    assert "deliver_item" in viz
    assert "get_item" in viz

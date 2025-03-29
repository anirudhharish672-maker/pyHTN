#!/usr/bin/env python3
"""
Comprehensive test for the HTN Planner.
This file tests all available planner functions with a simple delivery domain.
"""

import logging
import sys
from pprint import pprint

from pyhtn.conditions.conditions import AND
from pyhtn.conditions.fact import Fact
from pyhtn.domain.method import NetworkMethod
from pyhtn.domain.operators import NetworkOperator
from pyhtn.domain.task import GroundedTask, NetworkTask
from pyhtn.exceptions import FailedPlanException
from pyhtn.planner.planner import HtnPlanner
from pyhtn.domain.variable import V
from tests.blocks_world.blocks_world_env import BlocksWorldEnvironment


def setup_logging():
    """Configure logging to show detailed information."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)


def create_blocks_world_domain():
    """
    Create a simple blocks world domain with:
    - Blocks: block_a, block_b, block_c
    - Table: table

    The domain includes tasks for:
    - stack_blocks: Stack one block on another
    - clear_block: Make sure a block has nothing on top of it
    - move_block: Move a block from one location to another

    Methods:
    - stack_blocks_already_clear: Stack when both blocks are already clear
    - stack_blocks_need_clearing: Stack when target blocks need clearing first

    Operators:
    - pickup_block: Pick up a block from a location
    - putdown_block: Put down a block onto a location
    """
    print("Creating blocks world domain...")

    # Create variables for pattern matching
    block_var = V('block')
    source_var = V('source')
    dest_var = V('dest')
    clear_block_var = V('clear_block')
    type_var = V('type')

    # Define operators
    # Operator for picking up a block
    pickup_block_operator = NetworkOperator(
        name="pickup_block",
        effects=[],
        preconditions=AND(
            Fact(type="block", id=block_var, location=source_var, clear=True),
            Fact(type="hand", id="robot_hand", clear=True)
        ),
        args=[block_var, source_var],
        cost=1
    )

    # Operator for putting down a block
    putdown_block_operator = NetworkOperator(
        name="putdown_block",
        effects=[],
        preconditions=AND(
            Fact(type="block", id=block_var, location="hand"),
            Fact(type=type_var, id=dest_var, clear=True)
        ),
        args=[block_var, dest_var],
        cost=1
    )

    # Method for moving a block (already clear)
    move_block_method = NetworkMethod(
        name="move_block",
        subtasks=[
            pickup_block_operator,
            putdown_block_operator
        ],
        args=[block_var, source_var, dest_var],
        preconditions=AND(
            Fact(type="block", id=block_var, location=source_var, clear=True),
                Fact(type=type_var, id=dest_var, clear=True)
        )
    )


    # Method for clearing a block (when something is on it)
    clear_block_method = NetworkMethod(
        name="clear_block",
        subtasks=[
            NetworkTask("move_block", [clear_block_var, block_var, "table"])
        ],
        args=[block_var],
        preconditions=AND(
            Fact(type="block", id=clear_block_var, location=block_var, clear=True)
        )
    )

    # Method for directly clearing a block (when it's already clear)
    clear_block_already_method = NetworkMethod(
        name="clear_block",
        subtasks=[],  # No subtasks needed, it's already clear
        args=[block_var],
        preconditions=AND(
            Fact(type="block", id=block_var, clear=True)
        )
    )

    # Method for stacking blocks (when both blocks are clear)
    stack_blocks_already_clear_method = NetworkMethod(
        name="stack_blocks",
        subtasks=[
            NetworkTask("move_block", [block_var, source_var, dest_var])
        ],
        args=[block_var, source_var, dest_var],
        preconditions=AND(
            Fact(type="block", id=block_var, location=source_var, clear=True),
            Fact(type="block", id=dest_var, clear=True)
        )
    )

    # Method for stacking blocks (when blocks need clearing)
    stack_blocks_need_clearing_method = NetworkMethod(
        name="stack_blocks",
        subtasks=[
            NetworkTask("clear_block", [block_var]),
            NetworkTask("clear_block", [dest_var]),
            NetworkTask("move_block", [block_var, source_var, dest_var])
        ],
        args=[block_var, source_var, dest_var],
        preconditions=None  # No preconditions, we'll handle clearing as subtasks
    )

    # Define the domain structure
    domain = {
        "move_block/3": [move_block_method],
        "clear_block/1": [clear_block_method, clear_block_already_method],
        "stack_blocks/3": [stack_blocks_already_clear_method, stack_blocks_need_clearing_method],
    }

    print("Blocks world domain created successfully.")
    return domain


def create_blocks_initial_state():
    """Create the initial state for the blocks world problem."""
    print("Creating blocks world initial state...")
    state = [
        {"id": "block_a", "type": "block", "location": "table", "clear": True},
        {"id": "block_b", "type": "block", "location": "table", "clear": True},
        {"id": "block_c", "type": "block", "location": "table", "clear": True},
        {"id": "table", "type": "location", "clear": True},
        {"id": "robot_hand", "type": "hand", "clear": True}
    ]
    Fact(type="block", id='block_a', location='table', ),
    Fact(type="block", id='block_a', clear=True),
    Fact(type="location", id='block_b', clear=True)

    print("Initial blocks world state created successfully.")
    return state


def create_blocks_tasks():
    """Create the tasks for the blocks world planner."""
    print("Creating blocks world tasks...")
    tasks = [
        {"name": "stack_blocks", "arguments": ["block_a", "table", "block_b"], "priority": "high"},
        {"name": "stack_blocks", "arguments": ["block_c", "table", "block_a"], "priority": "medium"}
    ]
    print("Blocks world tasks created successfully.")
    return tasks





def test_apply_option(planner):
    """Test applying a method or operator to a task."""
    print("\n===== Testing Apply Option =====")

    # Set up a clean state
    planner.update_state(initial_state)

    try:
        # Get the next decomposition
        decomposition = planner.get_next_decomposition()
        task_node = decomposition['task_node']

        print(f"Applying option to task {task_node.id}...")
        result = planner.apply(task_node.id, 0)
        print("Option applied successfully.")
        print(f"Result: {result['node_type']} - {result['name']}")

        if result['node_type'] == 'method':
            print(f"Subtasks: {len(result['subtask_ids'])}")
        else:
            print(f"Effects: {result['effects']}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")


def test_planning():
    """Test the full planning process."""
    print("\n===== Testing Planning Process =====")
    planner = HtnPlanner(
        tasks=create_blocks_tasks(),
        domain=create_blocks_world_domain(),
        env=BlocksWorldEnvironment(),
        validate_input=True,
        logging=True,
        log_level=logging.DEBUG,
        console_output=True
    )
    planner.visualize()

    print("Starting planning process...", end="")
    _ = planner.plan()
    print("Plan generated successfully.")


    planner.print_current_plan()

    planner.print_current_trace()


def main():
    """Main function to run all tests."""
    print("Starting planner...")
    # setup_logging()

    # Test applying an option
    # test_apply_option(planner)

    # Test planning
    test_planning()

    # Test visualization
    # test_plan_visualization(planner)

    # Test tracking methods
    # test_plan_tracking(planner)

    print("\nAll tests completed successfully.")


if __name__ == "__main__":
    main()

import unittest
from unittest.mock import MagicMock, patch
import logging
import os
import time
from pyhtn.planner.planner import HtnPlanner
from pyhtn.domain.task import GroundedTask
from pyhtn.domain.method import NetworkMethod
from pyhtn.domain.operators import NetworkOperator
from pyhtn.exceptions import FailedPlanException, StopException

# Set up logging at the module level
log_dir = "test_logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = time.strftime('%Y%m%d_%H%M%S')
log_file_path = os.path.join(log_dir, f"planner_test_{timestamp}.log")

# Configure logger
logger = logging.getLogger('test_planner')
logger.setLevel(logging.DEBUG)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create file handler
file_handler = logging.FileHandler(log_file_path, mode='w')
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Print log file location
print(f"Log file created at: {os.path.abspath(log_file_path)}")


class TestPlannerDecompositionFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Called once before all tests in this class"""
        logger.info("=== TEST CLASS SETUP ===")
        logger.info(f"Logging to file: {os.path.abspath(log_file_path)}")

    @classmethod
    def tearDownClass(cls):
        """Called once after all tests in this class"""
        logger.info("=== TEST CLASS TEARDOWN ===")
        # Don't close handlers here - let Python handle it when the program exits

    def setUp(self):
        """Called before each test method"""
        logger.info("Setting up test case")

        # Mock environment
        self.env = MagicMock()
        self.env.get_state.return_value = [{'id': '1', 'type': 'location', 'name': 'kitchen'}]
        logger.info("Initialized mock environment")

        # Create a simple domain
        self.domain = {
            'make_coffee/0': [
                NetworkMethod(
                    name='make_coffee',
                    subtasks=[
                        NetworkOperator(name='get_mug', effects=[], args=[]),
                        NetworkOperator(name='add_coffee', effects=[], args=[]),
                        GroundedTask(name='heat_water', args=[]),
                        NetworkOperator(name='pour_water', effects=[], args=[]),
                    ],
                    preconditions=[]
                ),
                NetworkMethod(
                    name='make_coffee',
                    subtasks=[
                        NetworkOperator(name='get_cup', effects=[], args=[]),
                        NetworkOperator(name='add_instant_coffee', effects=[], args=[]),
                        NetworkOperator(name='boil_water', effects=[], args=[]),
                        NetworkOperator(name='pour_water', effects=[], args=[]),
                    ],
                    preconditions=[]
                )
            ],
            'heat_water/0': [
                NetworkMethod(
                    name='heat_water',
                    subtasks=[
                        NetworkOperator(name='fill_kettle', effects=[], args=[]),
                        NetworkOperator(name='turn_on_kettle', effects=[], args=[]),
                        NetworkOperator(name='wait_for_boil', effects=[], args=[]),
                    ],
                    preconditions=[]
                )
            ]
        }
        logger.info("Domain created with make_coffee and heat_water tasks")

        # Create a task
        self.tasks = [{'name': 'make_coffee', 'arguments': []}]
        logger.info("Task list created with 'make_coffee' task")

        logger.info("HTN Planner initialized")

        logger.info("Method and operator mocks configured")

    def tearDown(self):
        """Called after each test method"""
        logger.info("Test teardown complete - patches stopped")

    def get_new_planner(self):
        return HtnPlanner(
            tasks=self.tasks,
            domain=self.domain,
            env=self.env,
            enable_logging=True
        )

    def test_get_next_method_application_returns_first_method(self):
        """Test that get_next_method_application) returns the first applicable method"""
        logger.info("=== Starting test_get_next_method_application)_returns_first_method ===")

        planner = self.get_new_planner()

        # Print cursor state before
        logger.info("Cursor state before get_next_method_application): "
                         f"current_task={planner.cursor.current_task}, "
                         f"current_method_index={planner.cursor.current_method_index}")

        # Call get_next_method_application)
        task, method = planner.get_next_method_application()
        method = method[0]

        # Print results
        logger.info(f"get_next_method_application) returned task_name={task.name}, method.name={method.name}")
        logger.info(f"Cursor state after: current_method_index={planner.cursor.current_method_index}")

        # Verify results
        assert task.name == 'make_coffee'
        assert method.name == 'make_coffee'
        self.assertEqual(planner.cursor.current_method_index, 1)  # Should be incremented

    def test_get_next_method_application_returns_next_method(self):
        """Test that get_next_method_application) returns the next method when called again"""
        logger.info("=== Starting test_get_next_method_application)_returns_next_method ===")

        planner = self.get_new_planner()

        # Print cursor state before
        logger.info("Cursor state before first call: "
                         f"current_task={planner.cursor.current_task}, "
                         f"current_method_index={planner.cursor.current_method_index}")

        # Call get_next_method_application) twice
        task1, method1 = planner.get_next_method_application()
        method1 = method1[0]
        logger.info(f"First call returned task_name={task1.name}, method.name={method1.name}")
        logger.info(
            f"Cursor state after first call: current_method_index={planner.cursor.current_method_index}")

        task2, method2 = planner.get_next_method_application()
        method2 = method2[0]
        logger.info(f"Second call returned task_name={task2.name}, method.name={method2.name}")
        logger.info(
            f"Cursor state after second call: current_method_index={planner.cursor.current_method_index}")

        # Verify results
        assert task1.name == 'make_coffee'
        assert method1.name == 'make_coffee'

        assert task2.name == 'make_coffee'
        assert method2.name == 'make_coffee'

        self.assertEqual(planner.cursor.current_method_index, 2)  # Should be incremented again

    def test_get_next_method_application_returns_none_when_no_more_methods(self):
        """Test that get_next_method_application) returns None when no more methods are available"""
        logger.info("=== Starting test_get_next_method_application)_returns_none_when_no_more_methods ===")

        planner = self.get_new_planner()

        # Call get_next_method_application) twice
        task1, method1 = planner.get_next_method_application()
        method1 = method1[0]
        logger.info(f"First call returned task_name={task1.name}, method.name={method1.name}")

        task2, method2 = planner.get_next_method_application()
        method2 = method2[0]
        logger.info(f"Second call returned task_name={task2.name}, method.name={method2.name}")
        logger.info(
            f"Cursor state after second call: current_method_index={planner.cursor.current_method_index}")

        planner.cursor.print()

        task3, method3 = planner.get_next_method_application()

        assert task1.name == 'make_coffee'
        assert method1.name == 'make_coffee'

        assert task2.name == 'make_coffee'
        assert method2.name == 'make_coffee'

        assert task3.name == 'make_coffee'
        self.assertIsNone(method3)

        self.assertEqual(planner.cursor.current_method_index, 2)



    def test_apply_executes_operators_until_encountering_task(self):
        """Test that apply correctly executes operators until encountering a task"""
        logger.info("=== Starting test_apply_executes_operators_until_encountering_task ===")

        planner = self.get_new_planner()

        # Setup method to apply with operators and a subtask
        method_to_apply = self.domain['make_coffee/0'][0]  # This has 2 operators, then a task, then an operator
        logger.info(f"Using method {method_to_apply.name} with subtasks: " +
                         ", ".join([f"{t.name} ({t.type})" for t in method_to_apply.subtasks]))

        # Mock _execute_operator to always succeed
        with patch.object(planner, '_execute_operator', return_value=True) as mock_execute_op:
            logger.info("Mocked _execute_operator to always return True")

            # Mock plan method to just return the current plan
            with patch.object(planner, 'plan', return_value=planner.trace.get_current_plan()) as mock_plan:
                logger.info("Mocked plan method to return current plan")

                # Call apply
                logger.info("Calling apply with method_to_apply")
                result = planner.apply_method_application(GroundedTask('make_coffee', ()), method_to_apply)
                logger.info(f"apply returned result with {len(result) if result else 0} plan steps")

                # Print current state
                logger.info(f"Current cursor state: method={planner.cursor.current_method.name}, " +
                                 f"subtask_index={planner.cursor.current_subtask_index}")
                logger.info(f"Trace has {len(planner.trace.entries)} entries")

                # Verify results
                self.assertIsNotNone(result)  # Should return the current plan

                # Verify that the cursor is correctly set up
                self.assertEqual(planner.cursor.current_method, method_to_apply)
                self.assertEqual(planner.cursor.current_subtask_index, 0)

                # Verify that trace includes method application
                self.assertGreater(len(planner.trace.entries), 0)
                last_entry = planner.trace.entries[-1]
                logger.info(f"Last trace entry: type={last_entry.entry_type}, " +
                                 (f"method={last_entry.method.name}" if hasattr(last_entry, 'method') else "no method"))
                self.assertEqual(last_entry.entry_type, 'method')
                self.assertEqual(last_entry.method.name, 'make_coffee')

    def test_apply_with_no_tasks_in_subtasks(self):
        """Test apply when all subtasks are operators (no nested tasks)"""
        logger.info("=== Starting test_apply_with_no_tasks_in_subtasks ===")

        planner = self.get_new_planner()

        # Create a method with only operators
        method_with_only_operators = NetworkMethod(
            name='method_all_operators',
            subtasks=[
                NetworkOperator(name='op1', effects=[], args=[]),
                NetworkOperator(name='op2', effects=[], args=[]),
                NetworkOperator(name='op3', effects=[], args=[]),
            ],
            preconditions=[]
        )
        logger.info(f"Created method with only operators: {method_with_only_operators.name}")
        logger.info(f"Subtasks: " + ", ".join([t.name for t in method_with_only_operators.subtasks]))

        # Add to domain
        self.domain['all_operators/0'] = [method_with_only_operators]
        logger.info("Added method to domain")

        # Print trace state before
        logger.info(f"Trace entries before: {len(planner.trace.entries)}")

        # Mock _execute_operator to always succeed
        with patch.object(planner, '_execute_operator', return_value=True) as mock_execute_op:
            logger.info("Mocked _execute_operator to always return True")

            # Mock plan method
            with patch.object(planner, 'plan', return_value=[]) as mock_plan:
                logger.info("Mocked plan method to return empty list")

                # Call apply
                logger.info("Calling apply with method_with_only_operators")
                result = planner.apply_method_application(GroundedTask(name='all_operators', args=[]), method_with_only_operators)

                # Print trace state after
                logger.info(f"Trace entries after: {len(planner.trace.entries)}")

                # Find method entry in trace
                method_entry = next((e for e in planner.trace.entries if e.entry_type == 'method'), None)
                if method_entry:
                    logger.info(f"Found method entry in trace: {method_entry.method.name}")
                else:
                    logger.warning("No method entry found in trace")

                # Verify method is added to trace
                self.assertGreater(len(planner.trace.entries), 0)
                method_entry = next((e for e in planner.trace.entries if e.entry_type == 'method'), None)
                self.assertIsNotNone(method_entry)
                self.assertEqual(method_entry.method.name, 'method_all_operators')

    def test_apply_handles_backtracking(self):
        """Test that apply correctly handles backtracking when operators fail"""
        logger.info("=== Starting test_apply_handles_backtracking ===")

        planner = self.get_new_planner()

        # Create method with operators where one will fail
        method_with_failing_op = NetworkMethod(
            name='method_with_failure',
            subtasks=[
                NetworkOperator(name='op1', effects=[], args=[]),
                NetworkOperator(name='failing_op', effects=[], args=[]),  # This one will fail
                GroundedTask(name='subtask', args=[]),
            ],
            preconditions=[]
        )
        logger.info(f"Created method with failing operator: {method_with_failing_op.name}")
        logger.info(f"Subtasks: " + ", ".join([t.name for t in method_with_failing_op.subtasks]))

        # Add to domain
        self.domain['task_with_failure/0'] = [method_with_failing_op]
        logger.info("Added method to domain")

        # Mock _execute_operator to fail on the second operator
        def mock_execute_operator(operator):
            result = operator.name != 'failing_op'
            logger.info(f"Executing operator {operator.name}: {'Success' if result else 'FAILURE'}")
            return result

        with patch.object(planner, '_execute_operator', side_effect=mock_execute_operator) as mock_exec_op:
            # Mock _backtrack to return True
            with patch.object(planner, '_backtrack', return_value=True) as mock_backtrack:
                logger.info("Mocked _backtrack to return True")

                # Mock plan method
                with patch.object(planner, 'plan', side_effect=FailedPlanException("Test exception")) as mock_plan:
                    logger.info("Mocked plan method to raise FailedPlanException")

                    # Call apply should catch the exception
                    logger.info("Calling apply with method_with_failing_op")
                    try:
                        planner.apply_method_application(GroundedTask(name='task_with_failure', args=[]), method_with_failing_op)
                        logger.error("Expected exception was not raised")
                    except FailedPlanException as e:
                        logger.info(f"Caught expected exception: {str(e)}")

                    # Check that backtrack was called
                    logger.info(f"_backtrack was called: {mock_backtrack.called}")


def print_planner_state(self, planner):
    logger.info("===== PLANNER STATE =====")
    logger.info(f"Root tasks: {len(planner.root_tasks)}")
    if planner.root_tasks:
        for i, task in enumerate(planner.root_tasks):
            logger.info(f"  Task {i}: {task.name} (status: {task.status})")

    logger.info(f"Current task: {planner.cursor.current_task.name if planner.cursor.current_task else None}")
    logger.info(f"Current method: {planner.cursor.current_method.name if planner.cursor.current_method else None}")
    logger.info(f"Current subtask index: {planner.cursor.current_subtask_index}")
    logger.info(f"Available methods: {len(planner.cursor.available_methods)}")
    logger.info(f"Current method index: {planner.cursor.current_method_index}")
    logger.info(f"Stack depth: {len(planner.cursor.stack)}")

    logger.info(f"Trace entries: {len(planner.trace.entries)}")
    for i, entry in enumerate(planner.trace.entries[-5:]):  # Show last 5 entries
        logger.info(f"  Entry {i}: type={entry.type}")

    logger.info("=========================")


if __name__ == '__main__':
    unittest.main()

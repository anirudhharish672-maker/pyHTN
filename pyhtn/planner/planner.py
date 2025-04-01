from abc import abstractmethod
from copy import deepcopy
from itertools import chain
import logging
import time

from pyhtn.common.imports.typing import *
from pyhtn.domain.method import GroundedMethod
from pyhtn.domain.method import NetworkMethod
from pyhtn.domain.operators import GroundedOperator
from pyhtn.domain.operators import NetworkOperator
from pyhtn.domain.task import GroundedTask
from pyhtn.domain.task import NetworkTask
from pyhtn.exceptions import FailedPlanException, StopException
from pyhtn.validation import validate_domain, validate_tasks
from pyhtn.planner.planner_logger import PlannerLogger
from pyhtn.planner.trace import Trace


class Cursor:
    """
    Keeps track of the current position in the planning process.
    """

    def __init__(self):
        # Current task being processed
        self.current_task = None
        # Current method being applied to the task
        self.current_method = None
        # Index of current subtask in the method
        self.current_subtask_index = 0
        # Stack for backtracking [(task, method, subtask_index)]
        self.stack = []
        # Available methods for current task (to enable backtracking)
        self.available_methods = []
        # Index of current method in available_methods
        self.current_method_index = 0

    def set_task(self, task: Union[GroundedTask, NetworkTask]):
        """Set the current task being processed."""
        task.status = 'in-progress'
        self.current_task = task
        self.current_method = None
        self.current_subtask_index = 0
        self.available_methods = []
        self.current_method_index = 0

    def push_context(self):
        """Push current context to stack before moving to a subtask."""
        self.stack.append((
            self.current_task,
            self.current_method,
            self.current_subtask_index,
            self.available_methods,
            self.current_method_index
        ))

    def pop_context(self):
        """Pop back up to parent/previous context."""
        if not self.stack:
            return False  # Nothing to backtrack to
        (
            self.current_task,
            self.current_method,
            self.current_subtask_index,
            self.available_methods,
            self.current_method_index
        ) = self.stack.pop()

        # Moves on to next subtask.
        # If this function is being used for backtrack, the subtask index will be reset anyway
        self.current_subtask_index += 1

    def backtrack(self):
        """Restore previous context from stack."""
        if not self.pop_context():
            return False

        # Increment method index to try next method
        self.current_method_index += 1

        # Return False if no more methods to try
        if self.current_method_index >= len(self.available_methods):
            self.current_task.status = 'failed'
            return False

        # Set new method
        self.current_method = self.available_methods[self.current_method_index]
        self.current_subtask_index = 0
        return True

    def reset(self):
        """Reset cursor to initial state (clear all values)."""
        self.__init__()

    def print(self):
        """
        Print the current state of the cursor in a readable format.
        """
        print("\n========== CURSOR STATE ==========")

        if self.current_task:
            print(f"Current Task: {self.current_task.name} (Status: {self.current_task.status})")
        else:
            print("Current Task: None")

        if self.current_method:
            print(f"Current Method: {self.current_method.name}")
            print(f"Current Subtask Index: {self.current_subtask_index} of {len(self.current_method.subtasks)}")
        else:
            print("Current Method: None")
            print("Current Subtask Index: N/A")

        print(f"Available Methods: {len(self.available_methods)}")
        print(f"Current Method Index: {self.current_method_index}")
        print(f"Stack Depth: {len(self.stack)}")

        if self.stack:
            print("\nBacktrack Stack (most recent first):")
            for i, (task, method, subtask_idx, _, _) in enumerate(reversed(self.stack[-3:])):
                print(f"  {i}: Task '{task.name}', Method '{method.name if method else 'None'}', "
                      f"Subtask {subtask_idx}")

            if len(self.stack) > 3:
                print(f"  ... and {len(self.stack) - 3} more")

        print("==================================\n")


class TaskManager:
    def __init__(self, choice_criterion: str = 'random'):
        self.queue = []
        self.options = ()
        self.choice_criterion = choice_criterion

    def flatten(self):
        """
        Convert partially ordered task list into total ordered task list
        based on choice resolution criteria.
        """
        pass

    @abstractmethod
    def get_next_task(self):
        if not self.options:
            if not self.queue:
                return None
            self.options = tuple(self._get_next_tasks(self.queue))
        if self.options:
            if self.choice_criterion == 'random':
                pass


    def _get_next_tasks(self, tasks: Union[List, Tuple]) -> Union[List, Tuple]:
        """
        Returns list/tuple of tasks which no other task is constrained to precede.
        """
        if isinstance(tasks, list) and not isinstance(tasks[0], (list, tuple)):
            return list([tasks.pop()])
        elif isinstance(tasks, list):
            return self._get_next_tasks(tasks[0])
        elif isinstance(tasks, tuple):
            return tuple(
                chain.from_iterable(self._get_next_tasks(t) if isinstance(t, (list, tuple)) else (t,) for t in tasks))



class RootTaskQueue(TaskManager):
    def __init__(self, choice_criterion: str = 'random'):
        """

        :param choice_criterion: How partially ordered tasks should be selected. One of [random, ordered, cost]
        """
        super().__init__(choice_criterion)

    def get_next_task(self) -> GroundedTask:
        pass



class HtnPlanner:
    """
    This planner implements HTN (Hierarchical Task Network) planning using a graph of
    task, method, and operator nodes.
    """

    def __init__(self,
                 domain: Dict[str, List[NetworkMethod]],
                 tasks: List[dict] = None,
                 env=None,
                 validate_input: bool = False,
                 repeat_wait_time: int = 5,
                 order_by_cost: bool = False,
                 enable_logging: bool = False,
                 log_dir: str = None,
                 log_level: int = logging.INFO,
                 console_output: bool = False):
        """
        Initialize the HTN planner.
        :param tasks:
        :param domain:
        :param env:
        :param validate_input:
        :param repeat_wait_time:
        :param order_by_cost:
        :param enable_logging:
        :param log_dir:
        :param log_level:
        :param console_output:
        """
        # Validate inputs if requested
        if validate_input:
            validate_domain(domain)
            validate_tasks(tasks)

        # Initialize instance variables
        self.env = env
        self.state = self.env.get_state()
        self.root_tasks = []
        # Replace plan_list with trace
        self.trace = Trace()


        # Planning options
        self.validate = validate_input
        self.repeat_wait_time = repeat_wait_time
        self.order_by_cost = order_by_cost
        self.enable_logging = enable_logging

        # Setup logger
        if self.enable_logging:
            log_file = log_dir if log_dir is not None else 'planner_logs/planner_log.log'
            self.logger = PlannerLogger(log_file, log_level, console_output)
            self.logger.info("HTN Planner initialized")
        else:
            self.logger = None

        self.planning_start_time = None

        # Set domain network
        self.domain_network = domain

        # Create cursor to track planning state
        self.cursor = Cursor()

        if self.enable_logging:
            self.logger.info(f"Domain network built with {len(domain)} task types")
            self.logger.info(f"Domain network structure: {list(self.domain_network.keys())}")

        # Add tasks to planning queue
        if tasks is not None:
            self._add_tasks(tasks)

        if self.enable_logging:
            self.logger.info(f"Added {len(self.root_tasks)} root tasks")

    def reset(self):
        self.trace = Trace()
        self.cursor = Cursor()
        self.root_tasks = []

    def clear_tasks(self):
        """Clear all tasks."""
        self.root_tasks = []
        self.cursor.reset()

    def remove_task(self, task: str):
        """
        Remove the next occurrence of a task. If the provided task is the current task,
        it will be removed and the planner will automatically continue to the next text.
        """
        pass

    def remove_tasks(self, tasks: List[str]):
        """Remove all occurrences of the tasks in the given list."""
        pass

    def add_tasks(self, tasks: Union[dict, List[dict]]) -> None:
        """
        Add tasks to the planner.
        :param tasks: A list of task specifications.
        :return: None
        """
        if isinstance(tasks, dict):
            tasks = [tasks]
        self._add_tasks(tasks)

    def add_method(self,
                   task_name: str,
                   task_args: List['V'],
                   preconditions: 'Fact',
                   subtasks: List[Any]
                   ):
        new_method = NetworkMethod(name=task_name, args=task_args, preconditions=preconditions, subtasks=subtasks)
        domain_key = f"{task_name}/{len(task_args)}"
        self.domain_network[domain_key].append(new_method)
        return new_method



    def get_current_plan(self):
        """Gets the current plan."""
        return self.trace.get_current_plan()

    def get_current_trace(self, include_states: bool = False):
        """

        :param include_states: Whether to include state information in the output
        :return:
        """
        """Gets the current trace."""
        return self.trace.get_current_trace()

    def print_current_plan(self):
        """Prints the current plan."""
        return self.trace.print_plan()

    def print_current_trace(self, include_states: bool = False) -> None:
        """
        Prints the current trace.
        :param include_states: Whether to include state information in the output
        :return:
        """
        return self.trace.print_trace(include_states)

    def print_network(self, format: str = 'domain') -> None:
        """
        Visualize the planner network.
        :param format:
        :return: None
        """
        strings = []
        tab = '\t'
        header = "PLANNER NETWORK"
        border = '#' * (len(header) + 4)
        print(border)
        print('# ' + header + ' #')
        print(border + '\n')

        for task_key, methods in self.domain_network.items():
            task_name, num_args = task_key.split("/")
            strings.append(0 * tab + f'Task({task_name}, num_args={num_args})')

            for method in methods:
                strings.append(1 * tab + f'Method({method.name}, args={method.args})')
                for subtask in method.subtasks:
                    strings.append(2 * tab + f'{subtask.type.title()}({subtask.name}, args={subtask.args})')
        for s in strings:
            print(s)
        print('\n')

    def print_planner_state(self):
        print("===== PLANNER STATE =====")
        print(f"Root tasks: {len(self.root_tasks)}")
        if self.root_tasks:
            for i, task in enumerate(self.root_tasks):
                print(f"  Task {i}: {task.name} (status: {task.status})")

        print(f"Current task: {self.cursor.current_task.name if self.cursor.current_task else None}")
        print(f"Current method: {self.cursor.current_method.name if self.cursor.current_method else None}")
        print(f"Current subtask index: {self.cursor.current_subtask_index}")
        print(f"Available methods: {len(self.cursor.available_methods)}")
        print(f"Current method index: {self.cursor.current_method_index}")
        print(f"Stack depth: {len(self.cursor.stack)}")

        print(f"Trace entries: {len(self.trace.entries)}")
        for i, entry in enumerate(self.trace.entries[-5:]):  # Show last 5 entries
            print(f"  Entry {i}: type={entry.entry_type}")

        print("=========================")

    def plan(self, interactive: bool = False) -> List:
        """
        Execute the planning process.
        :return: The complete plan.
        """
        # Record start time
        self.planning_start_time = time.time()

        # Update state from environment
        if self.env and hasattr(self.env, 'get_state'):
            self.state = self.env.get_state()

        if self.enable_logging:
            self.logger.info("Starting automated planning")

        if not self.root_tasks:
            if self.enable_logging:
                self.logger.error("No tasks to plan for")
            raise StopException("There are no tasks to plan for")

        # TODO make sure using partial ordering logic to get next tasks
        # Main planning loop
        while self.root_tasks or self.cursor.current_task:
            # If we have a current task, process it
            if self.cursor.current_task:
                # If current task is completed, go back to previous context or get next task
                if self.cursor.current_task.status == 'succeeded':
                    if self.cursor.stack:
                        # Return to parent task and continue with next subtask
                        self.cursor.pop_context()

                        # Continue processing the method's subtasks
                        success = self._continue_method_execution()
                        if not success:
                            # If execution failed, try to backtrack
                            if not self._backtrack():
                                # If backtracking fails and we have root tasks, move to next root task
                                if self.root_tasks:
                                    self.cursor.current_task = None
                                else:
                                    # No more tasks to try, planning has failed
                                    if self.enable_logging:
                                        self.logger.error("No valid plan found")
                                    raise FailedPlanException("No valid plan found")
                        else:
                            if interactive:
                                return self.trace.get_current_plan()
                    else:
                        # Task completed at root level, move to next task
                        self.cursor.current_task = None
                elif self.cursor.current_task.status == 'failed':
                    # Try to backtrack to an alternative method
                    if not self._backtrack():
                        # If backtracking fails and we have root tasks, move to next root task
                        if self.root_tasks:
                            self.cursor.current_task = None
                        else:
                            # No more tasks to try, planning has failed
                            raise FailedPlanException(f"No valid plan found for task {self.cursor.current_task.name}")
                else:
                    # Process the current task
                    success = self._process_current_task()
                    if not success and not self._backtrack():
                        # If processing fails and backtracking fails
                        if self.root_tasks:
                            self.cursor.current_task = None
                        else:
                            raise FailedPlanException(f"No valid plan found for task {self.cursor.current_task.name}")
            else:
                # No current task, get the next task from root_tasks
                if self.root_tasks:
                    next_task = self.root_tasks.pop(0)
                    # Handle repeat tasks
                    if next_task.repeat > 0:
                        next_task.repeat -= 1
                        self.root_tasks.insert(0, next_task)

                    # Skip already completed tasks
                    if next_task.status == 'succeeded':
                        if self.enable_logging:
                            self.logger.info(f"Task {next_task.name} already succeeded, skipping")
                        continue

                    # Set as current task
                    self.cursor.set_task(next_task)
                else:
                    # No more tasks to process, planning is complete
                    break

        # All tasks processed successfully
        if self.enable_logging:
            plan = self.trace.get_current_plan()
            self.logger.info(f"Planning complete. Plan has {len(plan)} steps")
            plan_names = [op.name for op in plan]
            self.logger.info(f"Final plan: {', '.join(plan_names)}")

        return self.trace.get_current_plan()

    def get_next_method_application(self, all_methods: bool = False):
        """
        Steps to the next applicable method for a task.
        :param all_methods: Whether to return all methods or one method at a time
        :return: The current task and either next method or all methods
        """

        # Clear trace
        # self.trace = Trace()

        if not self.root_tasks:
            if self.enable_logging:
                self.logger.error("No tasks to plan for")
            raise StopException("There are no tasks to plan for. You must add tasks to the planner first"
                                " using add_tasks().")

        if not self.cursor.current_task or self.cursor.current_task.status == 'succeeded':
            if not self._set_cursor_to_new_task():
                return self.cursor.current_task, None

        if self.enable_logging:
            self.logger.info(f"Getting the next applicable method for task ({self.cursor.current_task.name})")

        # No more methods
        if self.cursor.current_method_index >= len(self.cursor.available_methods):
            if self.enable_logging:
                self.logger.info(f"All methods for task ({self.cursor.current_task.name}) have been returned."
                                  f" No remaining methods.")
            return self.cursor.current_task, None

        # Return all applicable methods
        if all_methods:
            # Set method index to end of list so that if called again, there is nothing to return
            self.cursor.current_method_index = len(self.cursor.available_methods)
            return self.cursor.current_task, self.cursor.available_methods


        # Get next method
        method = self.cursor.available_methods[self.cursor.current_method_index]
        self.cursor.current_method = method


        # Add to trace
        self.trace.add_method(
            self.cursor.current_task,
            method,
            f"Method #{self.cursor.current_method_index + 1} shown during interactive stepping"
        )

        # Advance to next method for next call
        self.cursor.current_method_index += 1

        if self.enable_logging:
            self.logger.info(f"Providing method: {method.name} for task {self.cursor.current_task.name}")

        return self.cursor.current_task, [method]

    def _set_cursor_to_new_task(self):
        next_task = self.root_tasks.pop(0)
        if self.enable_logging:
            self.logger.info(f"Moving on to new task: ({next_task.name})")
        # Handle repeat tasks
        if next_task.repeat > 0:
            next_task.repeat -= 1
            self.root_tasks.insert(0, next_task)
        self.cursor.set_task(next_task)
        # Get and store applicable methods
        methods = self._get_applicable_methods()
        if not methods:
            if self.enable_logging:
                self.logger.info(f"No applicable methods found for task: ({self.cursor.current_task.name})")
            return False

        self.cursor.available_methods = methods
        self.cursor.current_method_index = 0
        return True

    def apply_method_application(self, task: GroundedTask, method_to_apply):
        """
        Steps to the next subtask of a method.
        :param method_to_apply:
        :return: The method to execute
        """

        if not self.cursor.current_task:
            self.cursor.current_task = task
            self.cursor.available_methods.append(method_to_apply)
            self.cursor.current_method_index = 0
            self.cursor.current_method = method_to_apply
            self.cursor.current_subtask_index = 0

        # Add to trace
        self.trace.add_method(
            self.cursor.current_task,
            method_to_apply,
            f"Applying subtasks of method ({method_to_apply.name})"
        )

        method = None
        # Locate method in network
        for m in self.domain_network[self.cursor.current_task.domain_key]:
            if m.id == method_to_apply.id:
                method = m
                break

        # Set cursor to this method
        self.cursor.available_methods.append(method)
        self.cursor.current_method = method
        self.cursor.current_subtask_index = 0
        self.cursor.current_method_index = 0

        return self.plan(interactive=True)

    def _get_applicable_methods(self) -> List[GroundedMethod]:
        """Get methods applicable to the current task."""
        if self.enable_logging:
            self.logger.info(f"Locating applicable methods for task {self.cursor.current_task.name}")

        if not self.state:
            raise ValueError("State must be set before getting applicable methods")

        # Check if there are methods for this task type
        if self.cursor.current_task.domain_key not in self.domain_network:
            if self.enable_logging:
                self.logger.warning(f"No methods found for task {self.cursor.current_task.domain_key}")
            return []

        applicable_methods = []
        methods = self.domain_network[self.cursor.current_task.domain_key]

        # Sort by cost if requested
        if self.order_by_cost:
            methods = sorted(methods, key=lambda m: m.cost)

        # Check each method
        visited = []
        for method in methods:
            result = method.applicable(self.cursor.current_task, self.state, str(self.trace.get_current_plan()), visited)
            if result:
                applicable_methods.append(result)
                if self.enable_logging:
                    self.logger.info(f"Method {method.name} is applicable to task {self.cursor.current_task.name}")

        if self.enable_logging:
            self.logger.info(
                f"Found {len(applicable_methods)} applicable methods for task {self.cursor.current_task.name}")

        return applicable_methods

    def _add_tasks(self, tasks: List[Union[str, Tuple, Dict]]) -> None:
        """
        Internal method for adding tasks to the planning queue.
        :param tasks: A list of task specifications.
        :return: None
        """
        if self.validate:
            validate_tasks(tasks)

        for t in tasks:
            task_name = t.get('name')
            args = t.get('arguments', ())
            priority = t.get('priority', 'low')
            repeat = t.get('repeat', False)

            task = GroundedTask(task_name, args=args, priority=priority, repeat=repeat)
            self._add_task_node(task)

            if self.enable_logging:
                self.logger.info(f"Added task: ({task.name}) with priority ({priority}), repeat={repeat}")

    def _add_task_node(self, task: GroundedTask) -> None:
        """Add a task to the root tasks queue based on priority."""
        if task.priority == 'first' or not self.root_tasks:
            self.root_tasks.insert(0, task)
            if self.enable_logging:
                self.logger.info(f"Added task {task.name} as first task")
        else:
            priority_map = {'first': 0, 'high': 1, 'medium': 2, 'low': 3}
            task_priority = priority_map.get(task.priority)

            for i, existing_task in enumerate(self.root_tasks):
                existing_priority = existing_task.priority
                if isinstance(existing_priority, str):
                    existing_priority = priority_map.get(existing_priority, 3)

                if task_priority < existing_priority:
                    pos = i
                    self.root_tasks.insert(i, task)
                else:
                    pos = i + 1
                self.root_tasks.insert(pos, task)
                if self.enable_logging:
                    self.logger.info(f"Added task {task.name} at position {pos} with priority {task.priority}")
                return

            self.root_tasks.append(task)
            if self.enable_logging:
                self.logger.info(f"Added task {task.name} at the end with priority {task.priority}")

    def _process_current_task(self) -> bool:
        """Process the current task by finding and applying methods."""
        if self.enable_logging:
            self.logger.info(f"Processing task {self.cursor.current_task.name}")

        # Add task entry to trace
        self.trace.add_task(self.cursor.current_task)

        # If we already have a method, continue executing it
        if self.cursor.current_method:
            return self._continue_method_execution()

        # Get applicable methods for this task
        methods = self._get_applicable_methods()

        if not methods:
            if self.enable_logging:
                self.logger.warning(f"No applicable methods for task {self.cursor.current_task.name}")
            self.cursor.current_task.status = 'failed'
            return False

        # Store available methods for backtracking
        self.cursor.available_methods = methods
        self.cursor.current_method_index = 0
        self.cursor.current_method = methods[0]
        self.cursor.current_subtask_index = 0

        # Add method entry to trace
        self.trace.add_method(
            self.cursor.current_task,
            self.cursor.current_method,
            "Method selected as first applicable method"
        )

        if self.enable_logging:
            self.logger.info(
                f"Selected method {self.cursor.current_method.name} for task {self.cursor.current_task.name}")

        # Execute the selected method
        return self._continue_method_execution()

    def _continue_method_execution(self) -> bool:
        """Continue executing subtasks of the current method."""
        if not self.cursor.current_method:
            return False

        subtasks = self.cursor.current_method.subtasks

        if self.enable_logging:
            self.logger.info(f"Continuing method {self.cursor.current_method.name} execution "
                             f"from subtask {self.cursor.current_subtask_index+1} of {len(subtasks)}")

        # Process each subtask starting from current index
        while self.cursor.current_subtask_index < len(subtasks):
            subtask = subtasks[self.cursor.current_subtask_index]

            if isinstance(subtask, GroundedOperator):
                # Execute operator
                result = self._execute_operator(subtask)
                if not result:
                    return False
                # Move to next subtask
                self.cursor.current_subtask_index += 1


            elif isinstance(subtask, GroundedTask):
                if self.enable_logging:
                    self.logger.info(f"Encountered subtask {subtask.name}")

                # Add task entry to trace
                self.trace.add_task(
                    subtask,
                    parent_task=self.cursor.current_task,
                    parent_method=self.cursor.current_method
                )

                # Save current context before moving to subtask
                self.cursor.push_context()
                # Change current task to subtask
                self.cursor.set_task(subtask)

                # Return true to indicate we've moved to a subtask
                # and should continue from there
                return True

        # All subtasks completed successfully
        self.cursor.current_task.status = 'succeeded'
        if self.enable_logging:
            self.logger.info(f"All subtasks completed for task {self.cursor.current_task.name}")

        return True

    def _execute_operator(self, operator: NetworkOperator) -> bool:
        """Execute an operator and update state."""
        if self.enable_logging:
            self.logger.info(f"Executing operator {operator.name}")

        # Check if operator is applicable
        # Produces GroundedOperator
        result = operator.applicable(self.cursor.current_task, self.state)
        if not result:
            if self.enable_logging:
                self.logger.warning(f"Operator {operator.name} is not applicable")

            # Add failed operator to trace
            self.trace.add_operator(
                self.cursor.current_task,
                operator,
                False,
                state_before=deepcopy(self.state),
                state_after=None
            )

            return False

        # Execute in environment if available
        if self.env:
            try:
                state_before = deepcopy(self.state)

                if self.enable_logging:
                    self.logger.info(f"Executing operator {result.name} through agent")

                success = self.env.execute_action(
                    result.name,
                    result.args
                )

                # Update state from environment
                if hasattr(self.env, 'get_state'):
                    self.state = self.env.get_state()
                    state_after = self.state
                    if self.enable_logging:
                        self.logger.info(f"State updated from agent after operator execution")
                else:
                    state_after = deepcopy(self.state)

                # Add operator to trace
                self.trace.add_operator(
                    self.cursor.current_task,
                    result,
                    success,
                    state_before=state_before,
                    state_after=state_after
                )

                # Check execution result
                if not success:
                    if self.enable_logging:
                        self.logger.error(f"Operator {result.name} failed in environment")
                    self.cursor.current_task.status = 'failed'
                    return False

            except Exception as e:
                self.cursor.current_task.status = 'failed'

                # Add failed operator to trace
                self.trace.add_operator(
                    self.cursor.current_task,
                    result,
                    False,
                    state_before=deepcopy(self.state),
                    state_after=None
                )

                if self.enable_logging:
                    self.logger.error(f"Failed to execute operator {result.name}: {str(e)}")

                return False
        else:
            # No environment, just add to trace
            self.trace.add_operator(
                self.cursor.current_task,
                result,
                True,
                state_before=deepcopy(self.state),
                state_after=deepcopy(self.state)
            )

        if self.enable_logging:
            self.logger.info(f"Executing operator {result.name} was successful")
        return True

    def _backtrack(self) -> bool:
        """
        Attempt to backtrack to an alternative method.
        :return: Whether planner is able to backtrack to an alternative method.
        """
        if self.enable_logging:
            self.logger.info(f"Attempting to backtrack")

        # Try to backtrack with the cursor
        from_task = self.cursor.current_task
        result = self.cursor.backtrack()

        if result:
            # Add backtrack entry to trace
            self.trace.add_backtrack(
                from_task,
                self.cursor.current_task,
                f"Trying alternative method {self.cursor.current_method.name}"
            )

            # Add method entry for the new method
            self.trace.add_method(
                self.cursor.current_task,
                self.cursor.current_method,
                "Method selected after backtracking"
            )

            if self.enable_logging:
                self.logger.info(f"Backtracked to task {self.cursor.current_task.name}, "
                                 f"trying method {self.cursor.current_method.name}")
            return True
        else:
            if self.enable_logging:
                self.logger.warning("Backtracking failed, no alternative methods available")

            # Add backtrack entry to trace
            if self.cursor.stack:
                parent_task = self.cursor.stack[-1][0]
                self.trace.add_backtrack(
                    from_task,
                    parent_task,
                    "No alternative methods available"
                )
            else:
                self.trace.add_backtrack(
                    from_task,
                    None,
                    "No alternative methods available and no parent task"
                )

            return False

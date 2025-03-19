from copy import deepcopy
from itertools import chain
from random import choice
from time import sleep
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

from py_htn.domain import Task, Method, Operator, flatten
from py_htn.utils import removeTask
from py_htn.fact import Fact
from py_htn.conditions import AND
from py_htn.exceptions import FailedPlanException, StopException
from py_htn.validation import validate_domain, validate_state, validate_tasks

from py_htn.planner.plan_node import PlanNode
from py_htn.planner.plan_action import PlanAction
from py_htn.planner.planner_logger import PlannerLogger


class HtnPlanner:
    """
    A hierarchical task network planner that pre-processes domain information into a structured network.

    This planner implements HTN (Hierarchical Task Network) planning using a graph of
    task, method, and operator nodes. It provides methods for step-by-step planning
    as well as automatic planning.

    The domain is pre-processed to create method and operator nodes during initialization.
    Planning consists of traversing this network and creating task instances as needed.

    :param tasks: List of tasks to plan for
    :param domain: Domain definition containing methods and operators
    :param agent: Agent object with environment for executing actions
    :param validate_input: Whether to validate inputs
    :param repeat_wait_time: Time to wait before repeating a task
    :param order_by_cost: Whether to order options by cost
    :param logging: Whether to log planning steps
    :param log_dir: Directory for logging
    :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param console_output: Whether to also print logs to console

    :attribute domain_network: Pre-processed domain with method and operator nodes
    :attribute root_tasks: List of root task nodes to plan for
    :attribute state: Current state of the world
    :attribute plan_list: Executed plan actions
    :attribute plan_sequence_counter: Counter for plan action sequence IDs
    :attribute current_node: Current node being processed
    :attribute node_map: Map of node IDs to nodes
    :attribute task_instances: Map of task signatures to task instance nodes
    :attribute logger: Logger for tracking planner actions
    """

    def __init__(self,
                 tasks: List,
                 domain: Dict,
                 agent=None,
                 validate_input: bool = False,
                 repeat_wait_time: int = 5,
                 order_by_cost: bool = False,
                 logging: bool = False,
                 log_dir: str = None,
                 log_level: int = logging.INFO,
                 console_output: bool = False):
        """
        Initialize the HTN planner.

        :param tasks: List of tasks to plan for
        :param domain: Domain definition containing methods and operators
        :param agent: Agent object with environment for executing actions
        :param validate_input: Whether to validate inputs
        :param repeat_wait_time: Time to wait before repeating a task
        :param order_by_cost: Whether to order options by cost
        :param logging: Whether to log planning steps
        :param log_dir: Directory for logging
        :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        :param console_output: Whether to also print logs to console
        """
        # Validate inputs if requested
        if validate_input:
            validate_domain(domain)
            validate_tasks(tasks)

        # Initialize instance variables
        self.state = None
        self.root_tasks = []
        self.plan_list = []
        self.plan_sequence_counter = 0
        self.current_node = None
        self.node_map = {}
        self.task_instances = {}
        self.agent = agent

        # Planning options
        self.validate = validate_input
        self.repeat_wait_time = repeat_wait_time
        self.order_by_cost = order_by_cost
        self.enable_logging = logging

        # Setup logger
        if self.enable_logging:
            log_file = log_dir if log_dir is not None else 'logs/htn_planner.log'
            self.logger = PlannerLogger(log_file, log_level, console_output)
            self.logger.info("HTN Planner initialized")
        else:
            self.logger = None

        # Add tracking properties
        self.current_task_path = []  # List of task nodes from root to current
        self.execution_history = []  # History of executed operators and methods
        self.planning_depth = 0  # Current depth in the planning hierarchy
        self.last_decomposition = None  # Last decomposition result
        self.backtrack_points = []  # Available backtrack points

        # Build the domain network (pre-process domain)
        self.domain_network = self._build_domain_network(domain)

        if self.enable_logging:
            self.logger.info(f"Domain network built with {len(domain)} task types")
            self.logger.debug(f"Domain network structure: {list(self.domain_network.keys())}")

        # Create task nodes from the provided tasks
        self._add_tasks(tasks)

        if self.enable_logging:
            self.logger.info(f"Added {len(self.root_tasks)} root tasks")

    def _build_domain_network(self, domain: Dict) -> Dict:
        """
        Pre-process the domain to create a planning network structure.

        This creates nodes for all methods and operators in the domain during initialization,
        rather than creating them dynamically during planning.

        :param domain: Domain definition containing methods and operators
        :return: Structured domain network with nodes for methods and operators
        """
        network = {}

        # For each task type (key in domain)
        for task_signature, options in domain.items():
            network[task_signature] = {
                'methods': [],
                'operators': []
            }

            # Create nodes for methods and operators
            for option in options:
                if isinstance(option, Method):
                    method_node = PlanNode('method_template', option)
                    network[task_signature]['methods'].append(method_node)
                    self.node_map[method_node.id] = method_node

                    if self.enable_logging:
                        self.logger.debug(f"Added method template {option.name} for task {task_signature}")

                elif isinstance(option, Operator):
                    operator_node = PlanNode('operator_template', option)
                    network[task_signature]['operators'].append(operator_node)
                    self.node_map[operator_node.id] = operator_node

                    if self.enable_logging:
                        self.logger.debug(f"Added operator template {option.name} for task {task_signature}")

        return network

    def _add_tasks(self, tasks: List[Union[str, Tuple, Dict]]) -> None:
        """
        Converts a list of task specifications into Task objects and creates task nodes.

        :param tasks: List of tasks where each task can be:
                - str: Simple task name
                - tuple: (task_name, *args)
                - dict: {task: str/tuple,
                         priority: 'low'/'medium'/'high',
                         repeat: bool}
        """
        if self.validate:
            # First validate all tasks
            validate_tasks(tasks)

        for t in tasks:
            # Strings
            if isinstance(t, str):
                task = Task(t)
                self._add_task_node(task)

                if self.enable_logging:
                    self.logger.debug(f"Added task: {t}")

            # Dictionaries
            else:
                task_spec = t.get('task')
                args = t.get('arguments', ())
                priority = t.get('priority', 'low')
                repeat = t.get('repeat', False)

                task = Task(task_spec, args=args, priority=priority, repeat=repeat)
                self._add_task_node(task, priority=priority)

                if self.enable_logging:
                    self.logger.debug(f"Added task: {task_spec} with priority {priority}, repeat={repeat}")

    def _add_task_node(self, task: Task, priority: Union[str, int] = 'low') -> None:
        """
        Adds a task node to the root tasks.

        :param task: Task to add
        :param priority: Priority of the task ('first', 'high', 'medium', 'low' or 0-3)
        """
        task_node = PlanNode('task', task)
        self.node_map[task_node.id] = task_node

        # Store as a task instance
        task_signature = f"{task.name}/{len(task.args)}"
        if task_signature not in self.task_instances:
            self.task_instances[task_signature] = []
        self.task_instances[task_signature].append(task_node)

        # Add to root tasks based on priority
        if priority in [0, 'first']:
            self.root_tasks.insert(0, task_node)
            if self.enable_logging:
                self.logger.debug(f"Added task {task.name} as first task")
        elif not self.root_tasks:
            self.root_tasks.append(task_node)
            if self.enable_logging:
                self.logger.debug(f"Added task {task.name} as first root task")
        else:
            # Convert string priorities to integers
            priority_map = {'first': 0, 'high': 1, 'medium': 2, 'low': 3}
            task_priority = priority_map.get(priority, priority)

            # Insert based on priority
            for i, existing_node in enumerate(self.root_tasks):
                existing_priority = existing_node.content.priority
                if isinstance(existing_priority, str):
                    existing_priority = priority_map.get(existing_priority, 3)

                if task_priority <= existing_priority:
                    self.root_tasks.insert(i, task_node)
                    if self.enable_logging:
                        self.logger.debug(f"Added task {task.name} at position {i} with priority {priority}")
                    return

            # If we get here, add to the end
            self.root_tasks.append(task_node)
            if self.enable_logging:
                self.logger.debug(f"Added task {task.name} at the end with priority {priority}")

    def update_state(self, state: List[Dict]) -> None:
        """
        Updates the current state of the world.

        :param state: New state of the world
        """
        if self.validate:
            validate_state(state)

        old_state = self.state
        self.state = state

        if self.enable_logging:
            self.logger.debug(f"State updated. Old: {old_state}, New: {state}")

    def get_current_plan(self) -> List[PlanAction]:
        """
        Gets the current plan.

        :return: List of executed plan actions
        """
        return self.plan_list

    def get_applicable_methods(self, task_node: PlanNode) -> List[Dict]:
        """
        Gets all methods applicable to the given task.

        :param task_node: Task node to get methods for
        :return: List of applicable methods with their parameters
        """
        if not self.state:
            raise ValueError("State must be set before getting applicable methods")

        task = task_node.content
        domain_key = f"{task.name}/{len(task.args)}"

        if domain_key not in self.domain_network:
            if self.enable_logging:
                self.logger.warning(f"No methods found for task {domain_key}")
            return []

        applicable_methods = []
        method_nodes = self.domain_network[domain_key]['methods']

        # Sort by cost if requested
        if self.order_by_cost:
            method_nodes = sorted(method_nodes, key=lambda node: node.content.cost)

        # Check each method
        visited = []
        for method_node in method_nodes:
            method = method_node.content
            # Check if method is applicable
            result = method.applicable(task, self.state, str(self.plan_list), visited)
            if result:
                result['option'] = method
                result['template_node'] = method_node
                applicable_methods.append(result)

                if self.enable_logging:
                    self.logger.debug(f"Method {method.name} is applicable to task {task.name}")

        if self.enable_logging:
            self.logger.info(f"Found {len(applicable_methods)} applicable methods for task {task.name}")

        return applicable_methods

    def get_applicable_operators(self, task_node: PlanNode) -> List[Dict]:
        """
        Gets all operators applicable to the given task.

        :param task_node: Task node to get operators for
        :return: List of applicable operators with their parameters
        """
        if not self.state:
            raise ValueError("State must be set before getting applicable operators")

        task = task_node.content
        domain_key = f"{task.name}/{len(task.args)}"

        if domain_key not in self.domain_network:
            if self.enable_logging:
                self.logger.warning(f"No operators found for task {domain_key}")
            return []

        applicable_operators = []
        operator_nodes = self.domain_network[domain_key]['operators']

        # Sort by cost if requested
        if self.order_by_cost:
            operator_nodes = sorted(operator_nodes, key=lambda node: node.content.cost)

        # Check each operator
        for operator_node in operator_nodes:
            operator = operator_node.content
            # Check if operator is applicable
            result = operator.applicable(task, self.state)
            if result:
                result['option'] = operator
                result['template_node'] = operator_node
                applicable_operators.append(result)

                if self.enable_logging:
                    self.logger.debug(f"Operator {operator.name} is applicable to task {task.name}")

        if self.enable_logging:
            self.logger.info(f"Found {len(applicable_operators)} applicable operators for task {task.name}")

        return applicable_operators

    def get_next_decomposition(self, task_node_id: str = None) -> Dict:
        """
        Returns the current task and the next applicable method or operator.

        :param task_node_id: ID of the task node to get decompositions for.
                            If None, use the current node.
        :return: Dict containing the task node and list of applicable methods/operators
        :raises ValueError: If the task node doesn't exist
        :raises StopException: If there are no applicable methods or operators
        """
        # Get the task node
        if task_node_id is None:
            if self.current_node is None:
                if not self.root_tasks:
                    if self.enable_logging:
                        self.logger.error("No tasks available")
                    raise StopException("No tasks available")
                task_node = self.root_tasks[0]
            else:
                task_node = self.current_node
        else:
            task_node = self.node_map.get(task_node_id)
            if not task_node:
                if self.enable_logging:
                    self.logger.error(f"Task node with ID {task_node_id} not found")
                raise ValueError(f"Task node with ID {task_node_id} not found")

        # Set as current node
        self.current_node = task_node

        if self.enable_logging:
            self.logger.info(f"Getting decompositions for task {task_node.content.name}")

        # Get applicable methods and operators
        methods = self.get_applicable_methods(task_node)
        operators = self.get_applicable_operators(task_node)
        options = methods + operators

        if not options:
            task_node.status = 'failed'
            if self.enable_logging:
                self.logger.warning(f"No applicable methods or operators for task {task_node.content.name}")
            raise StopException(f"No applicable methods or operators for task {task_node.content.name}")

        if self.enable_logging:
            self.logger.info(f"Found {len(options)} decomposition options for task {task_node.content.name}")

        return {
            'task_node': task_node,
            'options': options
        }

    def apply(self, task_node_id: str, method_or_operator_index: int = 0) -> Dict:
        """
        Applies a method or executes an operator for a task.

        :param task_node_id: ID of the task node
        :param method_or_operator_index: Index of the method or operator to use
        :return: Result of the application
        :raises ValueError: If the task node doesn't exist or the method/operator index is invalid
        :raises StopException: If there are no applicable methods or operators
        """
        if self.enable_logging:
            self.logger.info(f"Applying decomposition to task {task_node_id}, option index {method_or_operator_index}")

        # Get task node and options
        decomposition = self.get_next_decomposition(task_node_id)
        task_node = decomposition['task_node']
        options = decomposition['options']

        if method_or_operator_index >= len(options):
            if self.enable_logging:
                self.logger.error(
                    f"Method/operator index {method_or_operator_index} is out of range (0-{len(options) - 1})")
            raise ValueError(f"Method/operator index {method_or_operator_index} is out of range (0-{len(options) - 1})")

        # Get the selected option
        selected = options[method_or_operator_index]
        option = selected['option']
        template_node = selected['template_node']

        if self.enable_logging:
            self.logger.info(f"Selected {option.__class__.__name__} {option.name}")

        # Apply the option
        return self._apply_option(task_node, selected, option, template_node)

    def plan(self, state: List[Dict], task_node_id: Optional[str] = None) -> List[PlanAction]:
        """
        Automatically plan from a specific task or from all root tasks.

        :param state: Initial state of the world
        :param task_node_id: ID of the task node to start from. If None, start from all root tasks.
        :return: Resulting plan
        :raises StopException: If there are no tasks to plan for
        :raises FailedPlanException: If planning fails
        """
        # Update state
        self.update_state(state)

        if self.enable_logging:
            self.logger.info("Starting automated planning")
            if task_node_id:
                self.logger.info(f"Planning from task node {task_node_id}")
            else:
                self.logger.info(f"Planning from all root tasks ({len(self.root_tasks)} tasks)")

        # Clear existing plan
        self.plan_list = []
        self.plan_sequence_counter = 0

        # Reset tracking info
        self._reset_tracking_info()

        # Start with the specified task or all root tasks
        tasks_to_process = []
        if task_node_id:
            task_node = self.node_map.get(task_node_id)
            if not task_node:
                if self.enable_logging:
                    self.logger.error(f"Task node with ID {task_node_id} not found")
                raise ValueError(f"Task node with ID {task_node_id} not found")
            tasks_to_process.append(task_node)
        else:
            tasks_to_process = self.root_tasks.copy()

        if not tasks_to_process:
            if self.enable_logging:
                self.logger.error("No tasks to plan for")
            raise StopException("There are no tasks to plan for")

        # Planning stack - stores backtracking points (tasks, state)
        stack = []

        # Process tasks until none are left
        while tasks_to_process:
            current_task = tasks_to_process.pop(0)
            current_task.status = 'in_progress'
            self.current_node = current_task

            if self.enable_logging:
                self.logger.info(f"Processing task {current_task.content.name}")

            # Update tracking
            self._update_tracking_info()

            # Skip tasks that have already been processed
            if current_task.status == 'succeeded':
                if self.enable_logging:
                    self.logger.debug(f"Task {current_task.content.name} already succeeded, skipping")
                continue

            # Get applicable methods and operators
            methods = self.get_applicable_methods(current_task)
            operators = self.get_applicable_operators(current_task)

            success = False

            # Try operators first (they're more specific than methods)
            for operator_result in operators:
                operator = operator_result['option']
                template_node = operator_result['template_node']

                if self.enable_logging:
                    self.logger.info(f"Trying operator {operator.name}")

                # Save current state for potential backtracking
                previous_state = deepcopy(self.state)

                try:
                    # Apply the operator
                    result = self._apply_option(current_task, operator_result, operator, template_node)

                    # If task repeats, reset it and add back to the queue
                    if current_task.content.repeat:
                        current_task.status = 'pending'
                        if self.enable_logging:
                            self.logger.info(f"Task {current_task.content.name} repeats, adding back to queue")
                        if self.repeat_wait_time > 0:
                            sleep(self.repeat_wait_time)
                        tasks_to_process.append(current_task)

                    success = True
                    break
                except Exception as e:
                    # If operator failed, restore state and try next
                    self.state = previous_state
                    if self.enable_logging:
                        self.logger.warning(f"Operator {operator.name} failed: {str(e)}")
                    continue

            # If no operator worked, try methods
            if not success and methods:
                method_result = methods[0]  # Choose the first applicable method
                method = method_result['option']
                template_node = method_result['template_node']

                if self.enable_logging:
                    self.logger.info(f"Trying method {method.name}")

                # Save current state to stack for potential backtracking
                stack.append((tasks_to_process.copy(), deepcopy(self.state)))
                self.backtrack_points.append(self.current_node.id if self.current_node else None)

                # Apply the method
                result = self._apply_option(current_task, method_result, method, template_node)

                # Add subtasks to the front of the queue
                if 'subtask_ids' in result:
                    new_subtasks = [self.node_map[node_id] for node_id in result['subtask_ids']]
                    tasks_to_process = new_subtasks + tasks_to_process
                    if self.enable_logging:
                        self.logger.info(f"Added {len(new_subtasks)} subtasks to processing queue")

                success = True

            # If no method or operator worked, backtrack
            if not success:
                current_task.status = 'failed'
                if self.enable_logging:
                    self.logger.warning(f"Task {current_task.content.name} failed, no applicable methods or operators")

                # If we have a stack, backtrack
                if stack:
                    if self.enable_logging:
                        self.logger.info("Backtracking to previous state")
                    self._backtrack(stack)
                else:
                    # No solution found
                    if self.enable_logging:
                        self.logger.error(f"No valid plan found for task {current_task.content.name}")
                    raise FailedPlanException(message=f"No valid plan found for task {current_task.content.name}")

        # All tasks processed successfully
        if self.enable_logging:
            self.logger.info(f"Planning complete. Plan has {len(self.plan_list)} steps")
            self.logger.debug(f"Final plan: {', '.join([a.name for a in self.plan_list])}")

            # Log the complete plan to a separate file if we're logging
            self._log_plan()

        return self.plan_list

    def _log_plan(self):
        """
        Log the complete plan to a separate file.
        """
        if not self.enable_logging:
            return

        plan_str = "==== PLAN ====\n"
        for idx, action in enumerate(self.plan_list):
            plan_str += f"Step {idx + 1}: {action.type} - {action.name}\n"
            plan_str += f"  Arguments: {[str(a) for a in action.args]}\n"
            if action.type == 'Method':
                plan_str += f"  Subtasks: {action.output}\n"
            else:
                plan_str += f"  Effects: {action.output}\n"
            plan_str += "\n"

        self.logger.info(plan_str)

    def _apply_option(self, task_node: PlanNode, selected_result: Dict,
                      option: Union[Method, Operator], template_node: PlanNode) -> Dict:
        """
        Core logic for applying a method or operator to a task.

        :param task_node: The task node to apply the option to
        :param selected_result: The result from get_applicable_methods/operators
        :param option: The method or operator to apply
        :param template_node: The template node for the method or operator
        :return: Result of the application
        """
        # Mark task as in progress
        task_node.status = 'in_progress'

        # Store common parameters
        matched_facts = selected_result['matched_facts']
        domain_key = f"{task_node.content.name}/{len(task_node.content.args)}"

        if isinstance(option, Method):
            # Create an instance of this method for this specific task
            method_node = PlanNode('method', option, parent=task_node, state=self.state)
            method_node.matched_facts = matched_facts
            task_node.add_child(method_node)
            self.node_map[method_node.id] = method_node

            if self.enable_logging:
                self.logger.info(f"Created method instance {option.name} for task {task_node.content.name}")

            # Add to plan
            self._add_plan_action(domain_key, option, self.state, matched_facts)

            # Create subtask nodes
            subtasks = selected_result['grounded_subtasks']
            if not isinstance(subtasks, (list, tuple)):
                subtasks = [subtasks]

            if self.enable_logging:
                self.logger.debug(f"Method {option.name} decomposed into {len(subtasks)} subtasks")

            subtask_ids = []
            for subtask in subtasks:
                subtask_node = PlanNode('task', subtask, parent=method_node, state=self.state)
                method_node.add_child(subtask_node)
                self.node_map[subtask_node.id] = subtask_node

                # Store as a task instance
                subtask_signature = f"{subtask.name}/{len(subtask.args)}"
                if subtask_signature not in self.task_instances:
                    self.task_instances[subtask_signature] = []
                self.task_instances[subtask_signature].append(subtask_node)

                subtask_ids.append(subtask_node.id)

                if self.enable_logging:
                    self.logger.debug(f"Created subtask {subtask.name} with ID {subtask_node.id}")

            method_node.status = 'succeeded'

            # Set current node to first subtask if it exists
            if method_node.children:
                self.current_node = method_node.children[0]
                if self.enable_logging:
                    self.logger.debug(f"Current node set to first subtask: {self.current_node.content.name}")

            # Record this decomposition
            self._record_decomposition(task_node, self.current_node, option, 'method')

            return {
                'node_id': method_node.id,
                'node_type': 'method',
                'name': option.name,
                'subtask_ids': subtask_ids,
                'matched_facts': matched_facts,
                'next_task_id': self.current_node.id if method_node.children else None
            }
        else:
            # Create an instance of this operator for this specific task
            operator_node = PlanNode('operator', option, parent=task_node, state=self.state)
            operator_node.matched_facts = matched_facts
            task_node.add_child(operator_node)
            self.node_map[operator_node.id] = operator_node

            if self.enable_logging:
                self.logger.info(f"Created operator instance {option.name} for task {task_node.content.name}")

            # Add to plan
            self._add_plan_action(domain_key, option, self.state, matched_facts)

            # Update state by executing the operator via the agent
            if self.agent:
                try:
                    if self.enable_logging:
                        self.logger.info(f"Executing operator {option.name} through agent")
                    self.agent.env.execute_action(option)
                    # Update our internal state from the agent
                    if hasattr(self.agent.env, 'get_state'):
                        prev_state = self.state
                        self.state = self.agent.env.get_state()
                        if self.enable_logging:
                            self.logger.debug(f"State updated from agent after operator execution")
                except Exception as e:
                    operator_node.status = 'failed'
                    task_node.status = 'failed'
                    if self.enable_logging:
                        self.logger.error(f"Failed to execute operator {option.name}: {str(e)}")
                    raise FailedPlanException(f"Failed to execute operator {option.name}: {str(e)}")

            operator_node.status = 'succeeded'
            task_node.status = 'succeeded'

            if self.enable_logging:
                self.logger.info(f"Operator {option.name} executed successfully")

            # If task repeats, reset it
            if task_node.content.repeat:
                task_node.status = 'pending'
                if self.enable_logging:
                    self.logger.debug(f"Task {task_node.content.name} marked for repetition")

            # Find next task to process, if any
            self._update_current_node_after_operator(task_node)

            # Record this execution
            self._record_decomposition(task_node, self.current_node, option, 'operator')

            return {
                'node_id': operator_node.id,
                'node_type': 'operator',
                'name': option.name,
                'effects': selected_result.get('effects', []),
                'matched_facts': matched_facts,
                'next_task_id': self.current_node.id if self.current_node != task_node else None
            }

    def _add_plan_action(self,
                         task: str,
                         action_object: Union[Operator, Method],
                         state: List[Dict],
                         matched_facts: AND,
                         backtrack: bool = False) -> None:
        """
        Adds an action to the plan.

        :param task: Task that triggered this action
        :param action_object: The action object (Method or Operator)
        :param state: State when this action was executed
        :param matched_facts: Facts that matched when applying this action
        :param backtrack: Whether we're backtracking
        """
        action = PlanAction(sequence_id=self.plan_sequence_counter,
                            task=task,
                            state=state,
                            action_object=action_object,
                            matched_facts=matched_facts)

        self.plan_list.append(action)
        self.plan_sequence_counter += 1

        if self.enable_logging:
            action_type = "Method" if isinstance(action_object, Method) else "Operator"
            self.logger.info(
                f"Added plan action: Step {self.plan_sequence_counter - 1}, {action_type} {action_object.name}")
            self.logger.log_plan_step(action) if hasattr(self.logger, 'log_plan_step') else None

    def _record_decomposition(self, from_node: PlanNode, to_node: Optional[PlanNode],
                              option: Union[Method, Operator], option_type: str) -> None:
        """
        Record a decomposition in the execution history.

        :param from_node: Source task node
        :param to_node: Target task node (could be None)
        :param option: Applied method or operator
        :param option_type: Type of option ('method' or 'operator')
        """
        from_name = from_node.content.name if hasattr(from_node.content, 'name') else str(from_node.content)
        to_name = to_node.content.name if to_node and hasattr(to_node.content, 'name') else (
            str(to_node.content) if to_node else None)

        self.last_decomposition = {
            'from_task': from_name,
            'to_task': to_name,
            'applied': option.name,
            'type': option_type
        }

        self.execution_history.append(self.last_decomposition)

        if self.enable_logging:
            self.logger.info(f"Recorded decomposition: {option_type} {option.name} from {from_name} to {to_name}")

        # Update tracking info
        self._update_tracking_info()

    def _backtrack(self, stack: List[Tuple]) -> None:
        """
        Perform backtracking logic.

        :param stack: Stack of backtracking points
        """
        tasks_to_process, state = stack.pop()

        # Get the node we're backtracking to
        backtrack_to = self.backtrack_points.pop() if self.backtrack_points else None

        if self.enable_logging:
            current_name = self.current_node.content.name if self.current_node else "None"
            backtrack_name = self.node_map[
                backtrack_to].content.name if backtrack_to and backtrack_to in self.node_map else "None"
            self.logger.info(f"Backtracking from {current_name} to {backtrack_name}")

        # Restore state
        self.state = state

        if backtrack_to and backtrack_to in self.node_map:
            self.current_node = self.node_map[backtrack_to]

        # Remove the last plan action
        if self.plan_list:
            removed_action = self.plan_list.pop()
            self.plan_sequence_counter -= 1

            if self.enable_logging:
                self.logger.info(f"Removed plan action: {removed_action.type} {removed_action.name}")

        # Update tracking after backtracking
        self._update_tracking_info()

        # Add backtracking info to execution history
        current_name = self.current_node.content.name if self.current_node and hasattr(
            self.current_node.content, 'name') else (
            str(self.current_node.content) if self.current_node else None)

        self.execution_history.append({
            'from_task': None,
            'to_task': current_name,
            'applied': 'backtrack',
            'type': 'control'
        })

        if self.enable_logging:
            self.logger.log_backtrack(None, current_name) if hasattr(self.logger, 'log_backtrack') else None

    def _update_current_node_after_operator(self, completed_task_node: PlanNode) -> None:
        """
        Updates the current node after an operator is executed.

        :param completed_task_node: The task node that was just completed
        """
        if self.enable_logging:
            self.logger.debug(f"Updating current node after operator for {completed_task_node.content.name}")

        # Use iterative approach to avoid recursion
        current = completed_task_node

        # If this task is part of a method, try to move to the next sibling
        if current.parent and current.parent.node_type == 'method':
            method_node = current.parent
            siblings = method_node.children

            for i, sibling in enumerate(siblings):
                if sibling.id == current.id and i + 1 < len(siblings):
                    # Move to next sibling
                    self.current_node = siblings[i + 1]

                    if self.enable_logging:
                        self.logger.debug(f"Moving to next sibling: {self.current_node.content.name}")

                    return

        # If no next sibling or no parent, try to find next pending task
        # Start by going up the tree until we find a task node's parent
        while current.parent:
            parent = current.parent

            # If parent is a method and its parent is a task
            if parent.node_type == 'method' and parent.parent and parent.parent.node_type == 'task':
                # Method is complete, mark parent task as complete too
                parent.parent.status = 'succeeded'

                if self.enable_logging:
                    self.logger.debug(
                        f"Method complete, marking parent task as succeeded: {parent.parent.content.name}")

                current = parent.parent
            else:
                current = parent

        # Now look for next root task that's not completed
        for task in self.root_tasks:
            if task.status == 'pending':
                self.current_node = task

                if self.enable_logging:
                    self.logger.debug(f"Moving to next pending root task: {task.content.name}")

                return

        # If no pending tasks, keep current node as is
        if self.enable_logging:
            self.logger.debug("No pending tasks remain, keeping current node")

    def _update_tracking_info(self) -> None:
        """
        Update internal tracking information - current path and planning depth.
        """
        # Clear current path
        self.current_task_path = []

        # Rebuild path from current node to root
        node = self.current_node
        while node:
            self.current_task_path.insert(0, node)
            node = node.parent

        # Update planning depth
        self.planning_depth = len(self.current_task_path) - 1

        if self.enable_logging:
            current_name = self.current_node.content.name if self.current_node else "None"
            self.logger.debug(f"Updated tracking info. Current node: {current_name}, depth: {self.planning_depth}")

    def _reset_tracking_info(self) -> None:
        """
        Reset all planning tracking information.
        """
        self.execution_history = []
        self.last_decomposition = None
        self.backtrack_points = []
        self.current_task_path = []
        self.planning_depth = 0

        if self.enable_logging:
            self.logger.debug("Reset all tracking information")

    def visualize(self, format='text') -> str:
        """
        Visualizes the planning graph.

        :param format: Format to use for visualization ('text', 'domain')
        :return: Visualization of the planning graph
        """
        if self.enable_logging:
            self.logger.info(f"Generating visualization in {format} format")

        if format == 'text':
            return self._visualize_text()
        elif format == 'domain':
            return self._visualize_domain()
        else:
            if self.enable_logging:
                self.logger.warning(f"Unsupported visualization format: {format}")
            return "Unsupported visualization format"

    def _visualize_text(self, node=None, depth=0) -> str:
        """
        Creates a text visualization of the planning graph.

        :param node: Node to start from. If None, start from root tasks.
        :param depth: Current depth in the graph
        :return: Text visualization of the planning graph
        """
        result = ""

        if node is None:
            # Start with root tasks
            for root_task in self.root_tasks:
                result += self._visualize_text_iterative(root_task)
        else:
            result += self._visualize_text_iterative(node)

        return result

    def _visualize_domain(self) -> str:
        """
        Creates a text visualization of the domain network.

        :return: Text visualization of the domain network
        """
        result = "Domain Network:\n"
        for task_signature, options in self.domain_network.items():
            result += f"Task: {task_signature}\n"

            result += "  Methods:\n"
            for method_node in options['methods']:
                result += f"    - {method_node.content.name}\n"

            result += "  Operators:\n"
            for operator_node in options['operators']:
                result += f"    - {operator_node.content.name}\n"

            result += "\n"

        return result

    def _visualize_text_iterative(self, start_node) -> str:
        """
        Non-recursive version of text visualization.

        :param start_node: Node to start from
        :return: Text visualization of the graph from start_node
        """
        result = ""
        stack = [(start_node, 0)]  # (node, depth)

        while stack:
            node, depth = stack.pop()

            # Add current node
            indent = "  " * depth
            result += f"{indent}- {str(node)}\n"

            # Add children (in reverse order to get correct order when popped)
            for child in reversed(node.children):
                stack.append((child, depth + 1))

        return result

    def get_current_method_trace(self) -> str:
        """
        Get the trace of method applications leading to the current state.

        :return: Formatted string showing method applications
        """
        trace = []
        for action in self.execution_history:
            if action['type'] == 'method':
                trace.append(f"{action['applied']}: {action['from_task']} → {action['to_task']}")

        if not trace:
            return "No methods applied yet"

        return "\n".join(trace)

    def get_current_operator_trace(self) -> str:
        """
        Get the trace of operator applications leading to the current state.

        :return: Formatted string showing operator applications
        """
        trace = []
        for action in self.execution_history:
            if action['type'] == 'operator':
                trace.append(f"{action['applied']}: {action['from_task']}")

        if not trace:
            return "No operators applied yet"

        return "\n".join(trace)

    def get_backtrack_points(self) -> List[Dict]:
        """
        Get information about available backtrack points.

        :return: List of dictionaries with information about each backtrack point
        """
        points = []
        for i, point_id in enumerate(self.backtrack_points):
            if point_id in self.node_map:
                node = self.node_map[point_id]
                node_name = node.content.name if hasattr(node.content, 'name') else str(node.content)
                points.append({
                    'index': i,
                    'node_id': point_id,
                    'task': node_name,
                    'node_type': node.node_type
                })

        if self.enable_logging:
            self.logger.debug(f"Retrieved {len(points)} backtrack points")

        return points

    def debug_log(self, message: str) -> None:
        """
        Add a debug log message to the execution history.

        :param message: Debug message to log
        """
        self.execution_history.append({
            'type': 'debug',
            'applied': 'log',
            'from_task': None,
            'to_task': None,
            'message': message
        })

        if self.enable_logging:
            self.logger.debug(f"Debug message: {message}")

    def get_planning_status(self) -> Dict:
        """
        Get the current planning status.

        :return: Dictionary with information about the current planning state
        """
        status = {
            'current_node': self.current_node.id if self.current_node else None,
            'current_task': self.current_node.content.name if self.current_node and hasattr(self.current_node.content,
                                                                                            'name') else None,
            'planning_depth': self.planning_depth,
            'task_path': [node.content.name if hasattr(node.content, 'name') else str(node.content) for node in
                          self.current_task_path],
            'last_action': self.last_decomposition,
            'execution_history_length': len(self.execution_history),
            'plan_length': len(self.plan_list),
            'backtrack_points': len(self.backtrack_points)
        }

        if self.enable_logging:
            self.logger.debug(
                f"Planning status retrieved: current task = {status['current_task']}, plan length = {status['plan_length']}")

        return status

    def get_task_hierarchy_string(self) -> str:
        """
        Get a string representation of the current task hierarchy.

        :return: Formatted string showing task hierarchy
        """
        if not self.current_task_path:
            return "No active task"

        result = ""
        for i, node in enumerate(self.current_task_path):
            indent = "  " * i
            status_marker = "→" if node == self.current_node else " "
            node_name = node.content.name if hasattr(node.content, 'name') else str(node.content)
            result += f"{indent}{status_marker} {node_name} ({node.status})\n"

        return result

    def get_execution_trace(self) -> str:
        """
        Get a trace of all executed actions.

        :return: Formatted string showing execution history
        """
        trace = "\n".join([
            f"{i + 1}. {action['type'].upper()}: {action['applied']} "
            f"({action['from_task']} → {action['to_task']})"
            for i, action in enumerate(self.execution_history)
        ])

        if self.enable_logging:
            self.logger.debug(f"Generated execution trace with {len(self.execution_history)} actions")

        return trace

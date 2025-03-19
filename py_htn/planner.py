
from copy import deepcopy


from json import dumps
from itertools import chain
from random import choice
from time import sleep
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union
from uuid import uuid4

from shop2.domain import Task, Axiom, Method, Operator, flatten
from shop2.utils import removeTask
from shop2.fact import Fact
from shop2.conditions import AND
from shop2.exceptions import FailedPlanException, StopException
from shop2.validation import validate_domain, validate_state, validate_tasks


class PlanNode:
    """
    A node in the planning graph representing a task, method, or operator.

    Attributes:
        id (str): Unique identifier for the node
        node_type (str): Type of node ('task', 'method', or 'operator')
        content (Union[Task, Method, Operator]): The object contained in the node
        parent (Optional[PlanNode]): Parent node in the graph
        children (List[PlanNode]): List of child nodes
        state (List[Dict]): State at this node
        matched_facts (Optional): Facts that matched when applying this node
        status (str): Status of the node ('pending', 'in_progress', 'succeeded', 'failed')
    """

    def __init__(self,
                 node_type: str,
                 content: Union[Task, Method, Operator],
                 parent: Optional['PlanNode'] = None,
                 state: List[Dict] = None):
        """
        Initialize a plan node.

        Args:
            node_type (str): Type of node ('task', 'method', or 'operator')
            content (Union[Task, Method, Operator]): The object contained in the node
            parent (Optional[PlanNode]): Parent node in the graph
            state (List[Dict]): State at this node
        """
        self.id = str(uuid4()).replace('-', '')
        self.node_type = node_type
        self.content = content
        self.parent = parent
        self.children = []
        self.state = state
        self.matched_facts = None
        self.status = 'pending'

    def add_child(self, node: 'PlanNode') -> None:
        """
        Add a child node to this node.

        Args:
            node (PlanNode): Node to add as a child
        """
        self.children.append(node)
        node.parent = self

    def __str__(self) -> str:
        if self.node_type == 'task':
            return f"Task: {self.content.name} ({self.status})"
        elif self.node_type == 'method':
            return f"Method: {self.content.name} ({self.status})"
        else:
            return f"Operator: {self.content.name} ({self.status})"

    def to_dict(self) -> Dict:
        """
        Convert node to dictionary for serialization.

        Returns:
            Dict: Dictionary representation of the node
        """
        return {
            'id': self.id,
            'node_type': self.node_type,
            'name': self.content.name if hasattr(self.content, 'name') else str(self.content),
            'status': self.status,
            'children': [child.id for child in self.children],
            'parent': self.parent.id if self.parent else None
        }


class PlanAction:
    """
    Represents a step in the executed plan.

    Attributes:
        id (int): Sequence ID in the plan
        action_id (str): ID of the action object
        task (str): Task that triggered this action
        type (str): Type of action ('Operator' or 'Method')
        output: Output of the action (effects or subtasks)
        name (str): Name of the action
        args: Arguments of the action
        preconditions: Preconditions of the action
        state (List[Dict]): State when this action was executed
        matched_facts: Facts that matched when applying this action
    """

    def __init__(self,
                 sequence_id: int,
                 task: str,
                 state: List[dict],
                 action_object: Union[Operator, Method],
                 matched_facts):
        """
        Initialize a plan action.

        Args:
            sequence_id (int): Sequence ID in the plan
            task (str): Task that triggered this action
            state (List[dict]): State when this action was executed
            action_object (Union[Operator, Method]): The action object
            matched_facts: Facts that matched when applying this action
        """
        self.id = sequence_id
        self.action_id = action_object.id
        self.task = task
        if isinstance(action_object, Operator):
            self.type = 'Operator'
            self.output = action_object.effects
        else:
            self.type = 'Method'
            self.output = [str(subtask) for subtask in action_object.subtasks]

        self.name = action_object.name
        self.args = action_object.args
        self.preconditions = action_object.preconditions
        self.state = deepcopy(state)
        self.matched_facts = matched_facts

    def __str__(self):
        return dumps(self.to_dict(), indent=2)

    def __repr__(self):
        return dumps(self.to_dict(), indent=2)

    def to_dict(self):
        """
        Convert action to dictionary for serialization.

        Returns:
            Dict: Dictionary representation of the action
        """
        return {
            'sequence_id': self.id,
            'node_type': self.type,
            'node_id': self.action_id,
            'node_name': self.name,
            'task': self.task,
            'args': [str(a) for a in self.args],
            'preconditions': '',
            'matched_facts': str(self.matched_facts)
        }


class NodePlanner:
    """
    A task-network planner that uses a node graph to represent the planning process.

    This planner implements HTN (Hierarchical Task Network) planning using a graph of
    task, method, and operator nodes. It provides methods for step-by-step planning
    as well as automatic planning.

    Attributes:
        domain (Dict): Domain definition containing methods and operators
        root_tasks (List[Task]): List of root tasks to plan for
        state (List[Dict]): Current state of the world
        plan (List[PlanAction]): Executed plan actions
        plan_sequence_counter (int): Counter for plan action sequence IDs
        current_node (Optional[PlanNode]): Current node being processed
        node_map (Dict[str, PlanNode]): Map of node IDs to nodes
        agent: Agent object with environment for executing actions
        validate (bool): Whether to validate inputs
        repeat_wait_time (int): Time to wait before repeating a task
        order_by_cost (bool): Whether to order options by cost
        logging (bool): Whether to log planning steps
        log_dir (str): Directory for logging
    """

    def __init__(self,
                 tasks: List,
                 domain: Dict,
                 agent=None,
                 validate_input: bool = False,
                 repeat_wait_time: int = 5,
                 order_by_cost: bool = False,
                 logging: bool = False,
                 log_dir: str = None):
        """
        Initialize the node planner.

        Args:
            tasks (List): List of tasks to plan for
            domain (Dict): Domain definition containing methods and operators
            agent: Agent object with environment for executing actions
            validate_input (bool, optional): Whether to validate inputs. Defaults to False.
            repeat_wait_time (int, optional): Time to wait before repeating a task. Defaults to 5.
            order_by_cost (bool, optional): Whether to order options by cost. Defaults to False.
            logging (bool, optional): Whether to log planning steps. Defaults to False.
            log_dir (str, optional): Directory for logging. Defaults to None.
        """
        # Validate inputs if requested
        if validate_input:
            validate_domain(domain)
            validate_tasks(tasks)

        # Initialize instance variables
        self.domain = domain
        self.root_tasks = []
        self.state = None
        self.plan = []
        self.plan_sequence_counter = 0
        self.current_node = None
        self.node_map = {}
        self.agent = agent

        # Planning options
        self.validate = validate_input
        self.repeat_wait_time = repeat_wait_time
        self.order_by_cost = order_by_cost
        self.logging = logging
        self.log_dir = log_dir if log_dir is not None else 'pyHTN-logger.log'

        # Add tracking properties
        self.current_task_path = []  # List of task nodes from root to current
        self.execution_history = []  # History of executed operators and methods
        self.planning_depth = 0  # Current depth in the planning hierarchy
        self.last_decomposition = None  # Last decomposition result
        self.backtrack_points = []  # Available backtrack points

        # Create task nodes from the provided tasks
        self._add_tasks(tasks)

    def _add_tasks(self, tasks: List[Union[str, Tuple, Dict]]) -> None:
        """
        Converts a list of task specifications into Task objects and creates task nodes.

        Args:
            tasks: List of tasks where each task can be:
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

            # Dictionaries
            else:
                task_spec = t['task']
                args = t.get('arguments', ())
                priority = t.get('priority', 'low')
                repeat = t.get('repeat', False)

                task = Task(task_spec, args=args, priority=priority, repeat=repeat)
                self._add_task_node(task, priority=priority)

    def _add_task_node(self, task: Task, priority: Union[str, int] = 'low') -> None:
        """
        Adds a task node to the root tasks.

        Args:
            task (Task): Task to add
            priority (Union[str, int], optional): Priority of the task. Defaults to 'low'.
        """
        task_node = PlanNode('task', task)
        self.node_map[task_node.id] = task_node

        # Add to root tasks based on priority
        if priority in [0, 'first']:
            self.root_tasks.insert(0, task_node)
        elif not self.root_tasks:
            self.root_tasks.append(task_node)
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
                    return

            # If we get here, add to the end
            self.root_tasks.append(task_node)

    def update_state(self, state: List[Dict]) -> None:
        """
        Updates the current state of the world.

        Args:
            state (List[Dict]): New state of the world
        """
        if self.validate:
            validate_state(state)
        self.state = state

    def get_current_plan(self) -> List[PlanAction]:
        """
        Gets the current plan.

        Returns:
            List[PlanAction]: List of executed plan actions
        """
        return self.plan

    def get_applicable_methods(self, task_node: PlanNode) -> List[Dict]:
        """
        Gets all methods applicable to the given task.

        Args:
            task_node (PlanNode): Task node to get methods for

        Returns:
            List[Dict]: List of applicable methods with their parameters
        """
        if not self.state:
            raise ValueError("State must be set before getting applicable methods")

        task = task_node.content
        domain_key = f"{task.name}/{len(task.args)}"

        applicable_methods = []
        options = self.domain.get(domain_key, [])

        # Sort by cost if requested
        if self.order_by_cost:
            options = sorted(options, key=lambda op: op.cost)

        # Check each method
        visited = []
        for option in options:
            if isinstance(option, Method):
                # Check if method is applicable
                result = option.applicable(task, self.state, str(self.plan), visited)
                if result:
                    result['option'] = option
                    applicable_methods.append(result)

        return applicable_methods

    def get_applicable_operators(self, task_node: PlanNode) -> List[Dict]:
        """
        Gets all operators applicable to the given task.

        Args:
            task_node (PlanNode): Task node to get operators for

        Returns:
            List[Dict]: List of applicable operators with their parameters
        """
        if not self.state:
            raise ValueError("State must be set before getting applicable operators")

        task = task_node.content
        domain_key = f"{task.name}/{len(task.args)}"

        applicable_operators = []
        options = self.domain.get(domain_key, [])

        # Sort by cost if requested
        if self.order_by_cost:
            options = sorted(options, key=lambda op: op.cost)

        # Check each operator
        for option in options:
            if isinstance(option, Operator):
                # Check if operator is applicable
                result = option.applicable(task, self.state)
                if result:
                    result['option'] = option
                    applicable_operators.append(result)

        return applicable_operators

    def get_next_decomposition(self, task_node_id: str = None) -> Dict:
        """
        Returns the current task and the next applicable method or operator.

        Args:
            task_node_id (str, optional): ID of the task node to get decompositions for.
                                         If None, use the current node. Defaults to None.

        Returns:
            Dict: Contains the task node and list of applicable methods/operators

        Raises:
            ValueError: If the task node doesn't exist
            StopException: If there are no applicable methods or operators
        """
        # Get the task node
        if task_node_id is None:
            if self.current_node is None:
                if not self.root_tasks:
                    raise StopException("No tasks available")
                task_node = self.root_tasks[0]
            else:
                task_node = self.current_node
        else:
            task_node = self.node_map.get(task_node_id)
            if not task_node:
                raise ValueError(f"Task node with ID {task_node_id} not found")

        # Set as current node
        self.current_node = task_node

        # Get applicable methods and operators
        methods = self.get_applicable_methods(task_node)
        operators = self.get_applicable_operators(task_node)
        options = methods + operators

        if not options:
            task_node.status = 'failed'
            raise StopException(f"No applicable methods or operators for task {task_node.content.name}")

        return {
            'task_node': task_node,
            'options': options
        }

    def apply(self, task_node_id: str, method_or_operator_index: int = 0) -> Dict:
        """
        Applies a method or executes an operator for a task.

        Args:
            task_node_id (str): ID of the task node
            method_or_operator_index (int, optional): Index of the method or operator to use. Defaults to 0.

        Returns:
            Dict: Result of the application

        Raises:
            ValueError: If the task node doesn't exist or the method/operator index is invalid
            StopException: If there are no applicable methods or operators
        """
        # Store previous state for tracking
        previous_node = self.current_node

        decomposition = self.get_next_decomposition(task_node_id)
        task_node = decomposition['task_node']
        options = decomposition['options']

        if method_or_operator_index >= len(options):
            raise ValueError(f"Method/operator index {method_or_operator_index} is out of range (0-{len(options) - 1})")

        # Get the selected option
        selected = options[method_or_operator_index]
        option = selected['option']

        # Mark task as in progress
        task_node.status = 'in_progress'

        # Create node for the selected option
        if isinstance(option, Method):
            # Create method node
            method_node = PlanNode('method', option, parent=task_node, state=self.state)
            method_node.matched_facts = selected['matched_facts']
            task_node.add_child(method_node)
            self.node_map[method_node.id] = method_node

            # Add to plan
            self._add_plan_action(f"{task_node.content.name}/{len(task_node.content.args)}",
                                  option, self.state, selected['matched_facts'])

            # Create subtask nodes
            subtasks = selected['grounded_subtasks']
            if not isinstance(subtasks, (list, tuple)):
                subtasks = [subtasks]

            for subtask in subtasks:
                subtask_node = PlanNode('task', subtask, parent=method_node, state=self.state)
                method_node.add_child(subtask_node)
                self.node_map[subtask_node.id] = subtask_node

            method_node.status = 'succeeded'

            # Set current node to first subtask if it exists
            if method_node.children:
                self.current_node = method_node.children[0]

            # Update tracking info
            self._update_tracking_info()

            # Record this decomposition
            task_name = task_node.content.name if hasattr(task_node.content, 'name') else str(task_node.content)
            current_task_name = self.current_node.content.name if hasattr(self.current_node.content, 'name') else str(
                self.current_node.content)

            self.last_decomposition = {
                'from_task': task_name,
                'to_task': current_task_name,
                'applied': option.name,
                'type': 'method'
            }

            self.execution_history.append(self.last_decomposition)

            return {
                'node_id': method_node.id,
                'node_type': 'method',
                'name': option.name,
                'subtask_ids': [node.id for node in method_node.children],
                'matched_facts': selected['matched_facts'],
                'next_task_id': self.current_node.id if method_node.children else None
            }
        else:
            # Create operator node
            operator_node = PlanNode('operator', option, parent=task_node, state=self.state)
            operator_node.matched_facts = selected['matched_facts']
            task_node.add_child(operator_node)
            self.node_map[operator_node.id] = operator_node

            # Add to plan
            self._add_plan_action(f"{task_node.content.name}/{len(task_node.content.args)}",
                                  option, self.state, selected['matched_facts'])

            # Update state by executing the operator via the agent
            if self.agent:
                try:
                    self.agent.env.execute_action(option)
                    # Update our internal state from the agent
                    if hasattr(self.agent.env, 'get_state'):
                        self.state = self.agent.env.get_state()
                except Exception as e:
                    operator_node.status = 'failed'
                    task_node.status = 'failed'
                    raise FailedPlanException(f"Failed to execute operator {option.name}: {str(e)}")

            operator_node.status = 'succeeded'
            task_node.status = 'succeeded'

            # If task repeats, reset it
            if task_node.content.repeat:
                task_node.status = 'pending'

            # Find next task to process, if any
            self._update_current_node_after_operator(task_node)

            # Update tracking info
            self._update_tracking_info()

            # Record this execution
            task_name = task_node.content.name if hasattr(task_node.content, 'name') else str(task_node.content)
            current_name = self.current_node.content.name if self.current_node and hasattr(self.current_node.content,
                                                                                           'name') else (
                str(self.current_node.content) if self.current_node else None)

            self.last_decomposition = {
                'from_task': task_name,
                'to_task': current_name,
                'applied': option.name,
                'type': 'operator'
            }

            self.execution_history.append(self.last_decomposition)

            return {
                'node_id': operator_node.id,
                'node_type': 'operator',
                'name': option.name,
                'effects': selected.get('effects', []),
                'matched_facts': selected['matched_facts'],
                'next_task_id': self.current_node.id if self.current_node != task_node else None
            }

    def _update_current_node_after_operator(self, completed_task_node):
        """
        Updates the current node after an operator is executed.

        Args:
            completed_task_node (PlanNode): The task node that was just completed
        """
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
                    return

        # If no next sibling or no parent, try to find next pending task
        # Start by going up the tree until we find a task node's parent
        while current.parent:
            parent = current.parent

            # If parent is a method and its parent is a task
            if parent.node_type == 'method' and parent.parent and parent.parent.node_type == 'task':
                # Method is complete, mark parent task as complete too
                parent.parent.status = 'succeeded'
                current = parent.parent
            else:
                current = parent

        # Now look for next root task that's not completed
        for task in self.root_tasks:
            if task.status == 'pending':
                self.current_node = task
                return

        # If no pending tasks, keep current node as is

    def step(self, task_node_id: str, method_or_operator_index: int = 0) -> Dict:
        """
        Steps into a specific method or operator for a task. 
        This is a legacy method, use apply() instead.

        Args:
            task_node_id (str): ID of the task node to step
            method_or_operator_index (int, optional): Index of the method or operator to use. Defaults to 0.

        Returns:
            Dict: Result of the step
        """
        return self.apply(task_node_id, method_or_operator_index)

    def plan(self, state: List[Dict], task_node_id: Optional[str] = None) -> List[PlanAction]:
        """
        Automatically plan from a specific task or from all root tasks.

        Args:
            state (List[Dict]): Initial state of the world
            task_node_id (Optional[str], optional): ID of the task node to start from. 
                                                   If None, start from all root tasks. Defaults to None.

        Returns:
            List[PlanAction]: Resulting plan

        Raises:
            StopException: If there are no tasks to plan for
            FailedPlanException: If planning fails
        """
        # Update state
        self.update_state(state)

        # Clear existing plan
        self.plan = []
        self.plan_sequence_counter = 0

        # Reset tracking info
        self.execution_history = []
        self.last_decomposition = None
        self.backtrack_points = []
        self.current_task_path = []
        self.planning_depth = 0

        # Start with the specified task or all root tasks
        tasks_to_process = []
        if task_node_id:
            task_node = self.node_map.get(task_node_id)
            if not task_node:
                raise ValueError(f"Task node with ID {task_node_id} not found")
            tasks_to_process.append(task_node)
        else:
            tasks_to_process = self.root_tasks.copy()

        if not tasks_to_process:
            raise StopException("There are no tasks to plan for")

        # Planning stack - stores backtracking points (tasks, state)
        stack = []

        # Process tasks until none are left
        while tasks_to_process:
            current_task = tasks_to_process.pop(0)
            current_task.status = 'in_progress'
            self.current_node = current_task

            # Update tracking
            self._update_tracking_info()

            # Skip tasks that have already been processed
            if current_task.status == 'succeeded':
                continue

            # Get applicable methods and operators
            methods = self.get_applicable_methods(current_task)
            operators = self.get_applicable_operators(current_task)

            success = False

            # Try operators first
            for operator_result in operators:
                operator = operator_result['option']

                # Store previous state for tracking
                previous_node = self.current_node

                # Create operator node
                operator_node = PlanNode('operator', operator, parent=current_task, state=self.state)
                operator_node.matched_facts = operator_result['matched_facts']
                current_task.add_child(operator_node)
                self.node_map[operator_node.id] = operator_node

                # Add to plan
                self._add_plan_action(f"{current_task.content.name}/{len(current_task.content.args)}",
                                      operator, self.state, operator_result['matched_facts'])

                # Update state by executing the operator via the agent
                if self.agent:
                    try:
                        self.agent.env.execute_action(operator)
                        # Update our internal state from the agent
                        if hasattr(self.agent.env, 'get_state'):
                            self.state = self.agent.env.get_state()
                    except Exception as e:
                        operator_node.status = 'failed'
                        current_task.status = 'failed'
                        if stack:
                            tasks_to_process, self.state = stack.pop()
                            self.backtrack_points.pop() if self.backtrack_points else None
                            if self.plan:
                                self.plan.pop()
                                self.plan_sequence_counter -= 1
                            continue  # Try next option
                        else:
                            raise FailedPlanException(f"Failed to execute operator {operator.name}: {str(e)}")

                operator_node.status = 'succeeded'
                current_task.status = 'succeeded'

                # Record this execution
                task_name = current_task.content.name if hasattr(current_task.content, 'name') else str(
                    current_task.content)

                self.last_decomposition = {
                    'from_task': task_name,
                    'to_task': task_name,  # Same for operators
                    'applied': operator.name,
                    'type': 'operator'
                }

                self.execution_history.append(self.last_decomposition)

                # If task repeats, reset it and add back to the queue
                if current_task.content.repeat:
                    current_task.status = 'pending'
                    if self.repeat_wait_time > 0:
                        sleep(self.repeat_wait_time)
                    tasks_to_process.append(current_task)

                success = True
                break

            # If no operator worked, try methods
            if not success and methods:
                method_result = methods[0]  # Choose the first applicable method
                method = method_result['option']

                # Store previous state for tracking
                previous_node = self.current_node

                # Save current state to stack for potential backtracking
                stack.append((tasks_to_process.copy(), deepcopy(self.state)))
                self.backtrack_points.append(self.current_node.id if self.current_node else None)

                # Create method node
                method_node = PlanNode('method', method, parent=current_task, state=self.state)
                method_node.matched_facts = method_result['matched_facts']
                current_task.add_child(method_node)
                self.node_map[method_node.id] = method_node

                # Add to plan
                self._add_plan_action(f"{current_task.content.name}/{len(current_task.content.args)}",
                                      method, self.state, method_result['matched_facts'])

                # Add subtasks to the front of the queue
                subtasks = method_result['grounded_subtasks']
                if not isinstance(subtasks, (list, tuple)):
                    subtasks = [subtasks]

                new_subtask_nodes = []
                for subtask in subtasks:
                    subtask_node = PlanNode('task', subtask, parent=method_node, state=self.state)
                    method_node.add_child(subtask_node)
                    self.node_map[subtask_node.id] = subtask_node
                    new_subtask_nodes.append(subtask_node)

                # Add subtasks to the front of the queue
                tasks_to_process = new_subtask_nodes + tasks_to_process

                method_node.status = 'succeeded'
                success = True

                # Record this decomposition
                task_name = current_task.content.name if hasattr(current_task.content, 'name') else str(
                    current_task.content)
                subtask_name = new_subtask_nodes[0].content.name if new_subtask_nodes and hasattr(
                    new_subtask_nodes[0].content, 'name') else (
                    str(new_subtask_nodes[0].content) if new_subtask_nodes else None)

                self.last_decomposition = {
                    'from_task': task_name,
                    'to_task': subtask_name,
                    'applied': method.name,
                    'type': 'method'
                }

                self.execution_history.append(self.last_decomposition)

                # Update current node and tracking
                if new_subtask_nodes:
                    self.current_node = new_subtask_nodes[0]
                    self._update_tracking_info()

            # If no method or operator worked, backtrack
            if not success:
                current_task.status = 'failed'

                # If we have a stack, backtrack
                if stack:
                    tasks_to_process, self.state = stack.pop()
                    backtrack_to = self.backtrack_points.pop() if self.backtrack_points else None
                    if backtrack_to and backtrack_to in self.node_map:
                        self.current_node = self.node_map[backtrack_to]

                    # Remove the last plan action
                    if self.plan:
                        self.plan.pop()
                        self.plan_sequence_counter -= 1

                    # Update tracking after backtracking
                    self._update_tracking_info()

                    # Add backtracking info to execution history
                    self.execution_history.append({
                        'from_task': current_task.content.name if hasattr(current_task.content, 'name') else str(
                            current_task.content),
                        'to_task': self.current_node.content.name if self.current_node and hasattr(
                            self.current_node.content, 'name') else (
                            str(self.current_node.content) if self.current_node else None),
                        'applied': 'backtrack',
                        'type': 'control'
                    })
                else:
                    # No solution found
                    raise FailedPlanException(message=f"No valid plan found for task {current_task.content.name}")

        # All tasks processed successfully
        return self.plan

    def _add_plan_action(self,
                         task: str,
                         action_object: Union[Operator, Method],
                         state: List[Dict],
                         matched_facts: AND,
                         backtrack: bool = False) -> None:
        """
        Adds an action to the plan.

        Args:
            task (str): Task that triggered this action
            action_object (Union[Operator, Method]): The action object
            state (List[Dict]): State when this action was executed
            matched_facts (AND): Facts that matched when applying this action
            backtrack (bool, optional): Whether we're backtracking. Defaults to False.
        """
        self.plan.append(PlanAction(sequence_id=self.plan_sequence_counter,
                                    task=task,
                                    state=state,
                                    action_object=action_object,
                                    matched_facts=matched_facts))
        self.plan_sequence_counter += 1

    def visualize(self, format='text'):
        """
        Visualizes the planning graph.

        Args:
            format (str, optional): Format to use for visualization. Defaults to 'text'.

        Returns:
            str: Visualization of the planning graph
        """
        if format == 'text':
            return self._visualize_text()
        else:
            return "Unsupported visualization format"

    def _visualize_text(self, node=None, depth=0):
        """
        Creates a text visualization of the planning graph.

        Args:
            node (Optional[PlanNode], optional): Node to start from. If None, start from root tasks. Defaults to None.
            depth (int, optional): Current depth in the graph. Defaults to 0.

        Returns:
            str: Text visualization of the planning graph
        """
        result = ""

        if node is None:
            # Start with root tasks
            for root_task in self.root_tasks:
                result += self._visualize_text_iterative(root_task)
        else:
            result += self._visualize_text_iterative(node)

        return result

    def _visualize_text_iterative(self, start_node):
        """
        Non-recursive version of text visualization.

        Args:
            start_node (PlanNode): Node to start from

        Returns:
            str: Text visualization of the graph from start_node
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

    def _update_tracking_info(self):
        """Update internal tracking information"""
        # Clear current path
        self.current_task_path = []

        # Rebuild path from current node to root
        node = self.current_node
        while node:
            self.current_task_path.insert(0, node)
            node = node.parent

        # Update planning depth
        self.planning_depth = len(self.current_task_path) - 1

    def get_current_method_trace(self):
        """
        Get the trace of method applications leading to the current state.

        Returns:
            str: Formatted string showing method applications
        """
        trace = []
        for action in self.execution_history:
            if action['type'] == 'method':
                trace.append(f"{action['applied']}: {action['from_task']} → {action['to_task']}")

        if not trace:
            return "No methods applied yet"

        return "\n".join(trace)

    def get_current_operator_trace(self):
        """
        Get the trace of operator applications leading to the current state.

        Returns:
            str: Formatted string showing operator applications
        """
        trace = []
        for action in self.execution_history:
            if action['type'] == 'operator':
                trace.append(f"{action['applied']}: {action['from_task']}")

        if not trace:
            return "No operators applied yet"

        return "\n".join(trace)

    def get_backtrack_points(self):
        """
        Get information about available backtrack points.

        Returns:
            List[Dict]: Information about each backtrack point
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

        return points

    def debug_log(self, message: str):
        """
        Add a debug log message to the execution history.

        Args:
            message (str): Debug message to log
        """
        self.execution_history.append({
            'type': 'debug',
            'applied': 'log',
            'from_task': None,
            'to_task': None,
            'message': message
        })

        def _update_tracking_info(self):
            """Update internal tracking information"""

        # Clear current path
        self.current_task_path = []

        # Rebuild path from current node to root
        node = self.current_node
        while node:
            self.current_task_path.insert(0, node)
            node = node.parent

        # Update planning depth
        self.planning_depth = len(self.current_task_path) - 1

    def get_planning_status(self):
        """
        Get the current planning status

        Returns:
            Dict: Information about the current planning state
        """
        return {
            'current_node': self.current_node.id if self.current_node else None,
            'current_task': self.current_node.content.name if self.current_node and hasattr(self.current_node.content,
                                                                                            'name') else None,
            'planning_depth': self.planning_depth,
            'task_path': [node.content.name if hasattr(node.content, 'name') else str(node.content) for node in
                          self.current_task_path],
            'last_action': self.last_decomposition,
            'execution_history_length': len(self.execution_history),
            'plan_length': len(self.plan),
            'backtrack_points': len(self.backtrack_points)
        }

    def get_task_hierarchy_string(self):
        """
        Get a string representation of the current task hierarchy

        Returns:
            str: Formatted string showing task hierarchy
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

    def get_execution_trace(self):
        """
        Get a trace of all executed actions

        Returns:
            str: Formatted string showing execution history
        """
        return "\n".join([
            f"{i + 1}. {action['type'].upper()}: {action['applied']} "
            f"({action['from_task']} → {action['to_task']})"
            for i, action in enumerate(self.execution_history)
        ])

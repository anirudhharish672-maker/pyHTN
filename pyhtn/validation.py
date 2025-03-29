from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


def validate_domain(domain):
    pass


def validate_state(state):
    pass


def validate_tasks(task_list: List[Union[str, Tuple, Dict]]) -> None:
    """
    Validates a list of tasks before conversion to Task objects.

    Args:
        task_list: List of tasks to validate

    Raises:
        ValueError: If any task format is invalid
        TypeError: If task_list contains invalid types
    """
    if not isinstance(task_list, list):
        raise TypeError("Input must be a list")

    for task_item in task_list:
        # Validate string format
        if isinstance(task_item, str):
            if not task_item:
                raise ValueError("Task string cannot be empty")

        # Validate tuple format
        elif isinstance(task_item, tuple):
            if not task_item:
                raise ValueError("Task tuple cannot be empty")
            if not isinstance(task_item[0], str):
                raise ValueError("First element of task tuple must be a string")

        # Validate dictionary format
        elif isinstance(task_item, dict):
            if 'name' not in task_item:
                raise ValueError("Dictionary task must contain 'name' key")

            task_spec = task_item['name']

            # Validate task specification in dictionary
            if isinstance(task_spec, str):
                if not task_spec:
                    raise ValueError("Task string in dictionary cannot be empty")
            else:
                raise ValueError(f"Task in dictionary must be string, got {type(task_spec)}")

            priority_levels = ['first', 'high', 'medium', 'low']
            # Validate priority if present
            if 'priority' in task_item and task_item['priority'] not in priority_levels:
                raise ValueError(
                    f"Invalid priority value: {task_item['priority']}. "
                    f"Must be one of: {priority_levels}"
                )
            # Validate repeat if present
            if 'repeat' in task_item and task_item['repeat'] not in [True, False]:
                raise ValueError(
                    f"Invalid repeat value: {task_item['repeat']}. "
                    f"Must be one of: {[True, False]}"
                )

        else:
            raise TypeError(f"Task must be string, tuple, or dictionary, got {type(task_item)}")
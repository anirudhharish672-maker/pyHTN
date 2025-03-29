from uuid import uuid4
from pyhtn.common.imports.typing import *

class BaseTask:
    """
    Base class for all task types in the HTN planner.
    
    :param name: Name of the task
    :param args: Arguments for the task
    """
    def __init__(self,
                 name: str,
                 args: Union[List[Any], Tuple[Any, ...]] = ()):
        self.type = 'task'
        self.id = str(uuid4()).replace('-', '')
        self.name = name
        self.args = args
        self.domain_key = f'{self.name}/{len(self.args)}'
        self.head = (self.name, *self.args)
        # index_chain keeps track of where task is in global task list for easier removal
        self.index_chain = None

    def __str__(self):
        return self._get_str()

    def __repr__(self):
        return self._get_str()

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id

    def _get_str(self):
        return f'BaseTask(name={self.name}, args={self.args})'




class GroundedTask(BaseTask):
    """
    Task with grounded arguments. Used to assign tasks to planner.

    :param name: Name of the task
    :param args: Arguments for the task
    :param priority: Priority of the task ('first', 'high', 'medium', 'low')
    :param repeat: Whether the task should repeat after completion
    """

    def __init__(self,
                 name: str,
                 args: Union[List[Any], Tuple[Any, ...]] = (),
                 priority: Union[int, str] = 'first',
                 repeat: bool = False):
        super().__init__(name, args)
        self.priority = priority
        self.repeat = repeat
        self.status = None

    def __str__(self):
        return self._get_str()

    def __repr__(self):
        return self._get_str()

    def _get_str(self):
        return f'GroundedTask(name={self.name}, args={self.args})'


class NetworkTask(BaseTask):
    """
    Represents a task in the network structure as a subtask of a method. Differs from GroundedTask above in that
    arguments need not be grounded.
    
    :param name: Name of the task
    :param args: Arguments for the task
    :param methods: List of methods for this task
    """
    def __init__(self,
                 name: str,
                 args: Union[List[Any], Tuple[Any, ...]] = (),
                 methods: List = None):
        super().__init__(name, args)
        self.methods = methods or []
        self.grounded_task_class = GroundedTask

    def __str__(self):
        return self._get_str()

    def __repr__(self):
        return self._get_str()

    def _get_str(self):
        return f'NetworkTask(name={self.name}, args={self.args})'


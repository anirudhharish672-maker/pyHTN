from pyhtn.common.imports.typing import *
from abc import ABC, abstractmethod
from pyhtn.htn.htn_elements import Task, Method, Operator, HTN_Element

from pyhtn.conditions.pattern_matching import (
    dict_to_tuple,
)

from py_plan.pattern_matching import build_index
from enum import Enum

class ExStatus(Enum):
    INITIALIZED : "ExStatus" = 1
    IN_PROGRESS : "ExStatus" = 2
    SUCCESS :     "ExStatus" = 3
    FAILED :      "ExStatus" = 4

class ElementExecution(ABC):
    def __init__(self,
                 element: HTN_Element,
                 state: List[dict],
                 match: Sequence[Any] = (),
                 parent_exec: Any = None,
                 child_execs: Sequence["ElementExecution"] = [],
                 ) -> None:

        self.element = element        
        self.state = state
        self.match = match
        self.parent_exec = parent_exec
        self.child_execs = child_execs
        self.status = ExStatus.INITIALIZED

    def _base_longhash(self):
        return unique_hash([
            self.element.id, self.match, self.parent_exec.id
        ])

    def _str_helper(self, kind):
        if(len(self.match) > 0):
            match_str = ", ".join([repr(x) for x in self.match])
            return f"{kind}({self.element.name!r}, {match_str})"
        else:
            return f"{kind}({self.element.name!r})"

class TaskEx(ElementExecution):
    def __init__(self,
                 task: Task,
                 state: List[dict],
                 match: Sequence[Any] = (),
                 parent_method_exec: Any = None,
                 child_method_execs: Sequence["MethodEx"] = [],                 
                 ) -> None:

        super().__init__(task, state, match,
            parent_method_exec, child_method_execs)

    @property
    def id(self):
        return f"TE_{self._base_longhash()}"

    @property
    def task(self):
        return self.element

    @property
    def parent_method_exec(self):
        return self.parent_exec

    @property
    def child_method_execs(self):
        return self.child_execs

    def __str__(self):
        return self._str_helper("TaskEx")

    __repr__ = __str__

    def get_child_executions(self, domain, state):
        ''' Get all method executions or operator execution down stream
            of this task execution.
        ''' 
        # from pyhtn.htn.element_executions import MethodEx, TaskEx
        task = self.task
        # key = f"{task.name}/{len(task.args)}"
        methods_or_op = domain[self.task.name]
        if(not isinstance(methods_or_op, list)):
            methods_or_op = [methods_or_op]

        child_execs = []
        for m_or_op in methods_or_op:
            # match_substs = m_or_op.get_match_substitutions(self, state, index)
            child_execs += m_or_op.get_match_executions(self, state)
        return child_execs

class MethodEx(ElementExecution):
    def __init__(self,
                 method: Method,
                 state: List[dict],
                 match: Sequence[Any] = (),
                 parent_task_exec: Any = None,
                 subtask_execs: Sequence[TaskEx] = [],
                 ) -> None:
        super().__init__(method, state, match,
            parent_task_exec, subtask_execs)

    @property
    def method(self):
        return self.element

    @property
    def parent_task_exec(self):
        return self.parent_exec

    @property
    def subtask_execs(self):
        return self.child_execs

    @property
    def id(self):
        return f"ME_{self._base_longhash()}"

    def __str__(self):
        return self._str_helper("MethodEx")

    __repr__ = __str__

class OperatorEx(ElementExecution):
    def __init__(self,
                 operator: Operator,
                 state: List[dict],
                 match: Sequence[Any] = (),
                 parent_task_exec: TaskEx = None,
                 ) -> None:
        super().__init__(operator, state, match, parent_task_exec)

    @property
    def id(self):
        return f"OE_{self._base_longhash()}"

    @property
    def operator(self):
        return self.element

    @property
    def parent_task_exec(self):
        return self.parent_exec

    def __str__(self):
        return self._str_helper("OperatorEx")

    __repr__ = __str__


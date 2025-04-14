from pyhtn.common.imports.typing import *
from abc import ABC, abstractmethod
from pyhtn.htn.htn_elements import HTN_Element

class ElementExecution(ABC):
    def __init__(self,
                 element: HTN_Element,
                 match: Sequence[Any] = (),
                 parent_exec: Any = None,
                 child_execs: Sequence[Any] = [],
                 ) -> None:

        self.element = element        
        self.match = match
        self.parent_exec = parent_exec
        self.child_execs = child_execs

    def _base_longhash(self):
        return unique_hash([
            self.element, self.match, self.parent_exec.id
        ])

    def _str_helper(self, kind):
        match_str = ", ".join([repr(x) for x in self.match])
        return f"{kind}({self.element.name!r}, {match_str})"


class TaskEx(ElementExecution):

    @property
    def id(self):
        return f"TE_{self._base_longhash()}"

    def __str__(self):
        return self._str_helper("TaskEx")

    __repr__ = __str__


class MethodEx(ElementExecution):
    @property
    def id(self):
        return f"ME_{self._base_longhash()}"

    def __str__(self):
        return self._str_helper("MethodEx")

    __repr__ = __str__

class OperatorEx(ElementExecution):
    @property
    def id(self):
        return f"OE_{self._base_longhash()}"

    def __str__(self):
        return self._str_helper("OperatorEx")

    __repr__ = __str__

    def 

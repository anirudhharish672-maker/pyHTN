from itertools import chain, permutations
from operator import or_
from typing import List, Tuple, Set, Dict, Union
from py_plan.unification import execute_functions

from py_htn.domain.task import NetworkTask, GroundedTask


# from py_htn.domain import BaseTask




def remove_task(T: Union[List, Tuple], task: Union[NetworkTask, GroundedTask]) -> Union[List, Tuple]:
    """
    Remove task from the list or tuple T.
    """
    if isinstance(T, list):
        return [result for t in T if (result := remove_task(t, task)) != task and result]
    elif isinstance(T, tuple):
        return tuple(result for t in T if (result := remove_task(t, task)) != task and result)
    else: 
        return T

def add_task(x: Union[List, Tuple], y: Union[List, Tuple]) -> Union[List, Tuple]:
    """
    Add task(s) x to the front of the list or tuple y.
    """
    if isinstance(y, list):
        return [x] + y
    elif isinstance(y, tuple):
        return (x,) + y
         
def replaceTask(T: Union[List, Tuple, BaseTask], task: BaseTask, ntask: BaseTask) -> Union[List, Tuple, BaseTask]:
    """
    Replace task with ntask in T.
    """
    if isinstance(T, list):
        return [result for t in T if (result := replaceTask(t, task, ntask)) != ntask or result == ntask]
    elif isinstance(T, tuple):
        return tuple(result for t in T if (result := replaceTask(t, task, ntask)) != ntask or result == ntask)
    else: 
        return ntask if T.head == task.head else T


def execute_functions(fun, s=()):
    """
    Traverses a fact executing any functions present within. Returns a fact
    where functions are replaced with the function return value. Allows to
    return with variable unlike `py_plan.unification.execute_functions`.
    """
    if s == ():
        s = {}

    if isinstance(fun, tuple) and len(fun) > 0:
        if fun[0] == or_:
            try:
                if execute_functions(fun[1], s) is not False:
                    return True
            except TypeError as e:
                if execute_functions(fun[2], s) is not False:
                    return True
                raise e
            return execute_functions(fun[2], s)

        if callable(fun[0]):
            return fun[0](*[execute_functions(ele, s) for ele in fun[1:]])
        else:
            return tuple(execute_functions(ele, s) for ele in fun)
    if fun in s:
        return execute_functions(s[fun])

    return fun


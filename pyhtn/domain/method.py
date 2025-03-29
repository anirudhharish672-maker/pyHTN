from random import choice
from .network_element import NetworkElement
from .operators import GroundedOperator
from .task import GroundedTask
from pyhtn.common.imports.typing import *
from pyhtn.conditions.pattern_matching import dict_to_tuple
from pyhtn.conditions.pattern_matching import fact_to_tuple
from pyhtn.conditions.pattern_matching import msubst
from pyhtn.conditions.pattern_matching import subst
from pyhtn.conditions.pattern_matching import tuples_to_dicts
from pyhtn.conditions.pattern_matching import unify
from py_plan.pattern_matching import build_index
from py_plan.pattern_matching import pattern_match



# Method class
class NetworkMethod(NetworkElement):
    def __init__(self,
                 name: str,
                 subtasks: List[Any],
                 args: Union[List[Any], Tuple[Any, ...]] = (),
                 preconditions=None,
                 cost=1) -> None:
        super().__init__(name, args, preconditions, cost)
        self.type = 'method'
        self.subtasks = subtasks

    def applicable(self, task, state, plan, visited):
        ptstate = dict_to_tuple(state)
        index = build_index(ptstate)
        substitutions = unify(task.head, self.head)
        if not self.preconditions:
            return GroundedMethod(name=self.name,
                                  subtasks=msubst(substitutions, self.subtasks),
                                  matched_facts=[],
                                  args=self.args,
                                  preconditions=self.preconditions,
                                  cost=self.cost,
                                  parent_id=self.id)
        ptconditions = fact_to_tuple(self.preconditions, variables=True)
        for ptcondition in ptconditions:
            if (task.name, str(ptcondition), str(state), plan) in visited:
                continue
            visited.append((task.name, str(ptcondition), str(state), plan))
            # Find if method's precondition is satisfied by state
            # Only checks one substitution, not all
            methods = [(self.name, theta) for theta in pattern_match(ptcondition, index, substitutions)]
            if methods:
                m, theta = choice(methods)
                grounded_subtask_args = msubst(theta, self.subtasks)
                grounded_subtasks = self._create_grounded_subtasks(grounded_subtask_args)
                matched_facts = tuples_to_dicts(subst(theta, tuple(ptcondition)), use_facts=True, use_and_operator=True)
                return GroundedMethod(name=self.name,
                                      subtasks=grounded_subtasks,
                                      matched_facts=matched_facts,
                                      args=self.args,
                                      preconditions=self.preconditions,
                                      cost=self.cost,
                                      parent_id=self.id)

        return False

    @staticmethod
    def _create_grounded_subtasks(grounded_subtask_args):
        grounded_subtasks = []
        for gsa in grounded_subtask_args:
            if gsa['type'] == 'operator':
                grounded_subtasks.append(GroundedOperator(name=gsa['name'], args=gsa['args'], effects=gsa['effects']))
            else:
                grounded_subtasks.append(GroundedTask(name=gsa['name'], args=gsa['args']))
        return grounded_subtasks

    def __str__(self):
        return self._get_str()

    def __repr__(self):
        return self._get_str()

    def _get_str(self):
        return f'NetworkMethod(name={self.name}, args={self.args}, subtasks={self.subtasks})'


class GroundedMethod(NetworkMethod):
    def __init__(self,
                 name: str,
                 subtasks: List[Any],
                 matched_facts: list[dict],
                 parent_id: str,
                 args: Union[List[Any], Tuple[Any, ...]] = (),
                 preconditions=None,
                 cost=1) -> None:
        super().__init__(name, subtasks, args, preconditions, cost)
        self.parent_id = parent_id
        self.matched_facts = matched_facts

    def __str__(self):
        return self._get_str()

    def __repr__(self):
        return self._get_str()

    def _get_str(self):
        return f'GroundedMethod(name={self.name}, args={self.args}, subtasks={self.subtasks})'
from itertools import chain
from random import choice
from pyhtn.common.imports.typing import *
from pyhtn.conditions.conditions import NOT
from pyhtn.conditions.fact import Fact
from pyhtn.conditions.pattern_matching import dict_to_tuple
from pyhtn.conditions.pattern_matching import fact_to_tuple
from pyhtn.conditions.pattern_matching import subst
from pyhtn.conditions.pattern_matching import tuples_to_dicts
from pyhtn.conditions.pattern_matching import unify
from pyhtn.domain.network_element import NetworkElement
from pyhtn.domain.variable import V
from py_plan.pattern_matching import build_index
from py_plan.pattern_matching import pattern_match
from py_plan.unification import execute_functions


class NetworkOperator(NetworkElement):
    def __init__(self,
                 name: str,
                 effects: List[Fact],
                 args: Union[List[Any], Tuple[Any, ...]] = (),
                 preconditions=None,
                 cost=1):
        # super().__init__(name, args, preconditions, cost)
        super().__init__(name, args, preconditions=preconditions, cost=cost)
        self.type = 'operator'
        self.effects = effects
        self.grounded_operator_class = GroundedOperator
        self.network_operator_class = NetworkOperator

        self.add_effects = set()
        self.del_effects = set()

        if isinstance(self.effects, Fact):
            self.add_effects.add(self.effects)
        elif isinstance(self.effects, NOT):
            self.del_effects.add(self.effects[0])
        else:
            for e in self.effects:
                if isinstance(e, NOT):
                    self.del_effects.add(e[0])
                else:
                    self.add_effects.add(e)

    def applicable(self, task: Any, state: list[dict[str, Any]]):
        # ptstate = fact2tuple(state, variables=False)[0]
        ptstate = dict_to_tuple(state)
        index = build_index(ptstate)
        substitutions = unify(task.head, self.head)
        if not self.preconditions:
            grounded_args = tuple([substitutions[f'?{v.name}'] if isinstance(v, V) and f'?{v.name}' in substitutions else v for v in self.args])
            return GroundedOperator(
                name=self.name,
                effects=self.effects,
                args=grounded_args,
                matched_facts=None,
                executed_effects=self.get_effects(substitutions),
                preconditions=self.preconditions,
                cost=self.cost,
                parent_id=self.id,
            )
        ptconditions = fact_to_tuple(self.preconditions, variables=True)
        for ptcondition in ptconditions:
            actions = [(self.name, theta) for theta in pattern_match(ptcondition, index, substitutions)]
            if actions:
                a, theta = choice(actions)
                grounded_args = tuple([theta[f'?{v.name}'] if isinstance(v, V) and f'?{v.name}' in theta else v for v in self.args])
                matched_facts = tuples_to_dicts(subst(theta, tuple(ptcondition)),
                                                     use_facts=True, use_and_operator=True)
                executed_effects = self.get_effects(theta)
                return GroundedOperator(
                    name=self.name,
                    effects=self.effects,
                    args=grounded_args,
                    matched_facts=matched_facts,
                    executed_effects=executed_effects,
                    preconditions=self.preconditions,
                    cost=self.cost,
                    parent_id=self.id,
                )
        return False

    def get_effects(self, theta):
        add_effects, del_effects = set(), set()
        for effect in self.add_effects:
            add_effects.add(effect.duplicate())
        for effect in self.del_effects:
            del_effects.add(effect.duplicate())

        all_effects = []
        for effect in chain(add_effects, del_effects):
            # print(effect)
            for cond in effect.conds:
                # print('cond: ', cond)
                if isinstance(cond.value, tuple) and callable(cond.value[0]):
                    cond.value = tuple([f'?{x.name}' if isinstance(x, V) else x for x in cond.value])
                effect[cond.attribute] = execute_functions(cond.value, theta) if not isinstance(cond.value, V) \
                    else subst(theta, cond.value)
            all_effects.append({k: effect[k] for k in effect})
        # return add_effects, delete_effects
        return all_effects

    def __str__(self):
        return self._get_str()

    def __repr__(self):
        return self._get_str()

    def _get_str(self):
        return f'NetworkOperator(name={self.name}, args={self.args})'


class GroundedOperator(NetworkOperator):
    def __init__(self,
                 name: str,
                 effects: List[Fact],
                 args: Union[List[Any], Tuple[Any, ...]] = (),
                 matched_facts: list[dict] = None,
                 executed_effects: List[dict] = None,
                 parent_id: str = None,
                 preconditions=None,
                 cost=1) -> None:
        super().__init__(name, effects, args, preconditions, cost)
        self.parent_id = parent_id
        self.executed_effects = executed_effects
        self.matched_facts = matched_facts

    def __str__(self):
        return self._get_str()

    def __repr__(self):
        return self._get_str()

    def _get_str(self):
        return f'GroundedOperator(name={self.name}, args={self.args})'
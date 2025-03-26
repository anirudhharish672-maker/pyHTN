from network_element import NetworkElement
from py_htn.conditions.pattern_matching import subst
from py_plan.pattern_matching import build_index
from py_plan.pattern_matching import pattern_match


class Axiom(NetworkElement):
    """
    An axiom class.
    """
    def __init__(self,
                 name: str,
                 preconditions=None) -> None:
        super().__init__(name, preconditions=preconditions)
        self.type = 'axiom'

    def applicable(self, state):
        index = build_index(state)
        for theta in pattern_match(self.preconditions, index): # Find if axiom's precondition is satisfied for state
            state.add(subst(theta, self.head))
        return state

    def __str__(self):
        s = f"Name: {self.name}\n"
        s += f"Conditions: {self.preconditions}\n"
        return s

    def __repr__(self):
        return f"<Axiom {self.name}>"
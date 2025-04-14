from dataclasses import dataclass
from uuid import uuid4


# variable_counter = 0


@dataclass(eq=True, frozen=True)
class Var:
    """
    A variable for pattern matching.
    """
    name: str

    def to_unify_str(self):
        return f"?{ self.name }"

    def __str__(self):
        return f"V('{self.name}')"

    __repr__ = __str__


def gen_variable():
    # global variable_counter
    # variable_counter += 1
    return V('genvar{}'.format(str(uuid4()).replace('-', '')[-8:]))


V = Var

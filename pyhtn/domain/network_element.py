from abc import ABC, abstractmethod
from uuid import uuid4
from pyhtn.common.imports.typing import *


class NetworkElement(ABC):
    """
    Abstract base class for network elements like Methods, Operators, and Axioms.

    :param name: The name of this element
    :param preconditions: Conditions that must be true for the element to be applicable
    :param cost: Cost of applying this element

    :attribute id: Unique identifier for the element
    :attribute name: The name of this element
    :attribute task: The corresponding task of the element
    :attribute args: Arguments of the element
    :attribute head: Tuple containing name and arguments
    :attribute preconditions: Conditions that must be true for the element to be applicable
    :attribute cost: Cost of applying this element
    """

    def __init__(self,
                 name: str,
                 args: Union[List[Any], Tuple[Any, ...]] = (),
                 preconditions=None,
                 cost=1) -> None:
        self.id = str(uuid4()).replace('-', '')
        self.name = name
        # self.task = None
        self.args = args
        self.head = (self.name, *self.args)
        self.preconditions = preconditions or []
        self.cost = cost

    @abstractmethod
    def applicable(self, *args, **kwargs):
        """
        Determines if this element is applicable in the given state.
        Must be implemented by subclasses.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def __str__(self):
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError('Subclasses must implement this method.')

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id
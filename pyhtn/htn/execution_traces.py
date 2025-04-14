from .htn_element import HTN_Element

class ElementExecution(ABC):
    def __init__(self,
                 htn_element: HTN_Element,
                 match: Sequence[Any] = ()) -> None:

        self.htn_element = htn_element        
        self.match = match





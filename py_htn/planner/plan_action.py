from copy import deepcopy
from json import dumps
from typing import Dict, List, Union

from py_htn.domain import Method, Operator


class PlanAction:
    """
    Represents a step in the executed plan.
    
    :param sequence_id: Sequence ID in the plan
    :param task: Task that triggered this action
    :param state: State when this action was executed
    :param action_object: The action object (Method or Operator)
    :param matched_facts: Facts that matched when applying this action
    
    :attribute id: Sequence ID in the plan
    :attribute action_id: ID of the action object
    :attribute task: Task that triggered this action
    :attribute type: Type of action ('Operator' or 'Method')
    :attribute output: Output of the action (effects or subtasks)
    :attribute name: Name of the action
    :attribute args: Arguments of the action
    :attribute preconditions: Preconditions of the action
    :attribute state: State when this action was executed
    :attribute matched_facts: Facts that matched when applying this action
    """
    
    def __init__(self,
                 sequence_id: int,
                 task: str,
                 state: List[dict],
                 action_object: Union[Operator, Method],
                 matched_facts):
        """
        Initialize a plan action.
        
        :param sequence_id: Sequence ID in the plan
        :param task: Task that triggered this action
        :param state: State when this action was executed
        :param action_object: The action object (Method or Operator)
        :param matched_facts: Facts that matched when applying this action
        """
        self.id = sequence_id
        self.action_id = action_object.id
        self.task = task
        if isinstance(action_object, Operator):
            self.type = 'Operator'
            self.output = action_object.effects
        else:
            self.type = 'Method'
            self.output = [str(subtask) for subtask in action_object.subtasks]

        self.name = action_object.name
        self.args = action_object.args
        self.preconditions = action_object.preconditions
        self.state = deepcopy(state)
        self.matched_facts = matched_facts

    def __str__(self):
        return dumps(self.to_dict(), indent=2)

    def __repr__(self):
        return dumps(self.to_dict(), indent=2)

    def to_dict(self):
        """
        Convert action to dictionary for serialization.
        
        :return: Dictionary representation of the action
        """
        return {
            'sequence_id': self.id,
            'node_type': self.type,
            'node_id': self.action_id,
            'node_name': self.name,
            'task': self.task,
            'args': [str(a) for a in self.args],
            'preconditions': '',
            'matched_facts': str(self.matched_facts)
        }

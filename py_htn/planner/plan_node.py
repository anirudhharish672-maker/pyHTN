from typing import Dict, List, Optional, Union
from uuid import uuid4

from py_htn.domain import Task, Method, Operator


class PlanNode:
    """
    A node in the planning graph representing a task, method, or operator.
    
    :param node_type: Type of node ('task', 'method', 'operator', 'method_template', 'operator_template')
    :param content: The object contained in the node (Task, Method, or Operator)
    :param parent: Parent node in the graph
    :param state: State at this node
    
    :attribute id: Unique identifier for the node
    :attribute node_type: Type of node
    :attribute content: The object contained in the node
    :attribute parent: Parent node in the graph
    :attribute children: List of child nodes
    :attribute state: State at this node
    :attribute matched_facts: Facts that matched when applying this node
    :attribute status: Status of the node ('pending', 'in_progress', 'succeeded', 'failed')
    """
    
    def __init__(self,
                 node_type: str,
                 content: Union[Task, Method, Operator],
                 parent: Optional['PlanNode'] = None,
                 state: List[Dict] = None):
        """
        Initialize a plan node.
        
        :param node_type: Type of node ('task', 'method', 'operator', 'method_template', 'operator_template')
        :param content: The object contained in the node (Task, Method, or Operator)
        :param parent: Parent node in the graph
        :param state: State at this node
        """
        self.id = str(uuid4()).replace('-', '')
        self.node_type = node_type
        self.content = content
        self.parent = parent
        self.children = []
        self.state = state
        self.matched_facts = None
        self.status = 'pending'

    def add_child(self, node: 'PlanNode') -> None:
        """
        Add a child node to this node.
        
        :param node: Node to add as a child
        """
        self.children.append(node)
        node.parent = self

    def __str__(self) -> str:
        if self.node_type == 'task':
            return f"Task: {self.content.name} ({self.status})"
        elif self.node_type in ['method', 'method_template']:
            return f"Method: {self.content.name} ({self.status})"
        else:
            return f"Operator: {self.content.name} ({self.status})"

    def to_dict(self) -> Dict:
        """
        Convert node to dictionary for serialization.
        
        :return: Dictionary representation of the node
        """
        return {
            'id': self.id,
            'node_type': self.node_type,
            'name': self.content.name if hasattr(self.content, 'name') else str(self.content),
            'status': self.status,
            'children': [child.id for child in self.children],
            'parent': self.parent.id if self.parent else None
        }

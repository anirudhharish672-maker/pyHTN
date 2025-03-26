from typing import Dict, List, Tuple, Any
from py_htn.domain.operators import GroundedOperator


class BlocksWorldEnvironment:
    """
    A simple environment class for the blocks world domain.

    This environment tracks the state of blocks, their locations, and allows
    executing actions corresponding to operators in the HTN planner.
    """

    def __init__(self, initial_state=None):
        """
        Initialize the blocks world environment.

        Args:
            initial_state: Optional initial state. If None, a default state is created.
        """
        if initial_state is None:
            self.state = self._create_default_state()
        else:
            self.state = initial_state

        # Define available actions (corresponding to operators)
        self.available_actions = {
            "pickup_block": self._pickup_block,
            "putdown_block": self._putdown_block
        }

    def _create_default_state(self) -> List[Dict]:
        """Create a default initial state with three blocks on the table."""
        return [
            {"id": "block_a", "type": "block", "location": "table", "clear": True},
            {"id": "block_b", "type": "block", "location": "table", "clear": True},
            {"id": "block_c", "type": "block", "location": "table", "clear": True},
            {"id": "table", "type": "location", "clear": True},
            {"id": "robot_hand", "type": "hand", "clear": True}
        ]

    def print_state(self):
        print('State: ')
        for f in self.get_state():
            print(f)

    def get_state(self) -> List[Dict]:
        """
        Get the current state of the blocks world.

        Returns:
            The current state as a list of fact dictionaries.
        """
        return self.state

    def execute_action(self, action: Any, arguments: Tuple = None) -> None:
        """
        Execute an action in the blocks world.

        Args:
            action: Either a string action name or an Operator object
            arguments: Tuple of arguments for the action (if action is a string)

        Raises:
            ValueError: If the action is not recognized or preconditions are not met
        """
        # Handle the case where action is an Operator object
        if isinstance(action, GroundedOperator):
            action_name = action.name
            # Extract arguments from the operator
            args = action.args
        else:
            # Handle the case where action is a string name
            action_name = action
            args = arguments

        if action_name not in self.available_actions:
            raise ValueError(f"Unknown action: {action_name}")

        # Execute the appropriate action
        return self.available_actions[action_name](*args)

    def _pickup_block(self, block_id: str, source_location: str) -> bool:
        """
        Pick up a block from a location.

        Args:
            block_id: ID of the block to pick up
            source_location: Location from which to pick up the block

        Raises:
            ValueError: If preconditions are not met
        """
        # Check preconditions
        block = self._find_fact("id", block_id, "type", "block")
        if not block:
            return False

        # Check if block is at the source location
        if block["location"] != source_location:
            return False

        # Check if block is clear
        if not block.get("clear", False):
            return False

        # Check if hand is empty
        hand = self._find_fact("id", "robot_hand", "type", "hand")
        if not hand.get("clear", False):
            return False

        # Update the state
        # 1. Update block location
        block["location"] = "hand"

        # 2. Mark source as clear (if it's not the table)
        if source_location != "table":
            source = self._find_fact("id", source_location)
            if source:
                source["clear"] = True

        # 3. Mark hand as not clear
        hand["clear"] = False
        return True

    def _putdown_block(self, block_id: str, destination_location: str) -> bool:
        """
        Put down a block onto a location.

        Args:
            block_id: ID of the block to put down
            destination_location: Location where to put the block

        Raises:
            ValueError: If preconditions are not met
        """
        # Check preconditions
        block = self._find_fact("id", block_id, "type", "block")
        if not block:
            return False

        # Check if block is in hand
        if block["location"] != "hand":
            return False

        # Check if destination is clear
        destination = None
        if destination_location == "table":
            # Table is always considered clear for placing
            pass
        else:
            destination = self._find_fact("id", destination_location)
            if not destination:
                return False
            if not destination.get("clear", False):
                return False

        # Update the state
        # 1. Update block location
        block["location"] = destination_location

        # 2. Mark destination as not clear (unless it's the table)
        if destination_location != "table" and destination:
            destination["clear"] = False

        # 3. Mark block as clear
        block["clear"] = True

        # 4. Mark hand as clear
        hand = self._find_fact("id", "robot_hand", "type", "hand")
        hand["clear"] = True
        return True

    def _find_fact(self, key: str, value: Any, *args) -> Dict:
        """
        Find a fact in the state that matches the given key-value pairs.

        Args:
            key: The first key to match
            value: The value for the first key
            *args: Additional key-value pairs (flattened)

        Returns:
            The matching fact or None if not found
        """
        # Build pairs from args
        pairs = [(key, value)]

        # Process additional args in pairs
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                pairs.append((args[i], args[i + 1]))

        # Search for matching fact
        for fact in self.state:
            match = True
            for k, v in pairs:
                if k not in fact or fact[k] != v:
                    match = False
                    break

            if match:
                return fact

        return None

    def __str__(self) -> str:
        """Return a string representation of the environment state."""
        result = "Blocks World State:\n"

        # Find all blocks
        blocks = [fact for fact in self.state if fact.get("type") == "block"]

        # Show blocks and their locations
        for block in blocks:
            clear_status = "clear" if block.get("clear", False) else "not clear"
            result += f"  {block['id']} is on {block['location']} and is {clear_status}\n"

        # Show hand status
        hand = self._find_fact("id", "robot_hand")
        if hand:
            hand_status = "empty" if hand.get("clear", False) else "holding something"
            result += f"  Robot hand is {hand_status}\n"

        return result
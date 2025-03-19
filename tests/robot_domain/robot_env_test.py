from shop2.domain import Task, Method, Operator
from shop2.fact import Fact
from shop2.conditions import AND, OR, NOT, Filter
from shop2.common import V
from tabulate import tabulate

# Import the NodePlanner
from planner2 import NodePlanner, PlanNode


# Simple agent class for demonstration
class RobotAgent:
    def __init__(self):
        self.env = RobotEnvironment()


class RobotEnvironment:
    def __init__(self):
        self.state = []
        self.locations = ["kitchen", "living_room", "bedroom"]
        self.items = {}  # Map of item name to location
        self.robot_location = None
        self.robot_holding = None

    def execute_action(self, operator):
        """
        Execute an operator and update the state by modifying existing facts.

        Args:
            operator: The operator to execute
        """
        print(f"Executing operator: {operator.name}")

        # Handle based on operator type
        if operator.name == 'move':
            from_loc = operator.args[0].name if hasattr(operator.args[0], 'name') else operator.args[0]
            to_loc = operator.args[1].name if hasattr(operator.args[1], 'name') else operator.args[1]

            # Find robot fact and update its location
            for fact in self.state:
                if fact.get('type') == 'robot':
                    print(f"  Moving robot from {fact['location']} to {to_loc}")
                    fact['location'] = to_loc
                    self.robot_location = to_loc

        elif operator.name == 'pickup':
            item = operator.args[0].name if hasattr(operator.args[0], 'name') else operator.args[0]
            item_location = None

            # Find the item's location
            for fact in self.state:
                if fact.get('type') == 'item' and fact.get('name') == item:
                    item_location = fact.get('location')
                    # Remove the item from the state since it's being picked up
                    self.state.remove(fact)
                    break

            # Update robot to hold the item
            for fact in self.state:
                if fact.get('type') == 'robot':
                    print(f"  Robot picking up {item} from {item_location}")
                    fact['holding'] = item
                    self.robot_holding = item
                    if 'gripper' in fact:
                        del fact['gripper']

            # Update items tracking
            if item in self.items:
                del self.items[item]

        elif operator.name == 'drop':
            item = operator.args[0].name if hasattr(operator.args[0], 'name') else operator.args[0]
            robot_location = None

            # Find robot location and update holding status
            for fact in self.state:
                if fact.get('type') == 'robot':
                    robot_location = fact.get('location')
                    if fact.get('holding') == item:
                        print(f"  Robot dropping {item} at {robot_location}")
                        del fact['holding']
                        fact['gripper'] = 'empty'
                        self.robot_holding = None

            # Add item back to the state at the robot's location
            if robot_location:
                new_item_fact = Fact(id=f"i-{item}", type="item", name=item, location=robot_location)
                self.state.append(new_item_fact)
                self.items[item] = robot_location

        # Update internal state tracking
        self._update_internal_state()

        # Render the environment after action
        self.render()

    def _update_internal_state(self):
        """Update internal state tracking from facts"""
        for fact in self.state:
            if fact.get('type') == 'robot':
                self.robot_location = fact.get('location')
                self.robot_holding = fact.get('holding')
            elif fact.get('type') == 'item':
                self.items[fact.get('name')] = fact.get('location')

    def render(self):
        """Render the environment in ASCII format"""
        print("\n" + "=" * 50)
        print("ENVIRONMENT STATE:")
        print("=" * 50)

        # Create a grid for locations
        grid = []
        for location in self.locations:
            # Determine what's in this location
            location_contents = []

            # Check if robot is here
            if self.robot_location == location:
                if self.robot_holding:
                    location_contents.append(f"🤖 Robot holding {self.robot_holding}")
                else:
                    location_contents.append("🤖 Robot")

            # Check if any items are here
            for item, item_loc in self.items.items():
                if item_loc == location:
                    location_contents.append(f"📦 {item}")

            if not location_contents:
                location_contents = ["(empty)"]

            grid.append([location, "\n".join(location_contents)])

        print(tabulate(grid, headers=["Location", "Contents"], tablefmt="grid"))
        print("=" * 50 + "\n")

    def get_state(self):
        """Get the current state"""
        return self.state


# Example domain definition with Task objects
domain = {
    "move/2": [
        Operator(
            head=("move", V("from_loc"), V("to_loc")),
            preconditions=AND(
                Fact(type="location", name=V("from_loc")),
                Fact(type="location", name=V("to_loc")),
                Fact(type="robot", location=V("from_loc"))
            ),
            effects=AND(
                Fact(type="robot", location=V("to_loc")),
                NOT(Fact(type="robot", location=V("from_loc")))
            )
        )
    ],
    "pickup/1": [
        Operator(
            head=("pickup", V("item")),
            preconditions=AND(
                Fact(type="item", name=V("item"), location=V("loc")),
                Fact(type="robot", location=V("loc")),
                Fact(type="robot", gripper="empty")
            ),
            effects=AND(
                Fact(type="robot", holding=V("item")),
                NOT(Fact(type="robot", gripper="empty")),
                NOT(Fact(type="item", name=V("item"), location=V("loc")))
            )
        )
    ],
    "drop/1": [
        Operator(
            head=("drop", V("item")),
            preconditions=AND(
                Fact(type="robot", holding=V("item")),
                Fact(type="robot", location=V("loc"))
            ),
            effects=AND(
                Fact(type="item", name=V("item"), location=V("loc")),
                Fact(type="robot", gripper="empty"),
                NOT(Fact(type="robot", holding=V("item")))
            )
        )
    ],
    "get_item/1": [
        Method(
            head=("get_item", V("item")),
            preconditions=AND(
                Fact(type="item", name=V("item"), location=V("item_loc")),
                Fact(type="robot", location=V("robot_loc"))
            ),
            subtasks=[
                Task(name="move", args=(V("robot_loc"), V("item_loc"))),
                Task(name="pickup", args=(V("item"),))
            ]
        )
    ],
    "deliver_item/2": [
        Method(
            head=("deliver_item", V("item"), V("dest")),
            preconditions=AND(
                Fact(type="item", name=V("item")),
                Fact(type="location", name=V("dest"))
            ),
            subtasks=[
                Task(name="get_item", args=(V("item"),)),
                Task(name="move", args=(V("curr_loc"), V("dest"))),
                Task(name="drop", args=(V("item"),))
            ]
        )
    ]
}

# Initial state
initial_state = [
    Fact(id="l1", type="location", name="kitchen"),
    Fact(id="l2", type="location", name="living_room"),
    Fact(id="l3", type="location", name="bedroom"),
    Fact(id="r1", type="robot", location="kitchen", gripper="empty"),
    Fact(id="i1", type="item", name="book", location="living_room")
]

# Tasks to plan for
tasks = [
    {"task": "deliver_item", "arguments": ("book", "bedroom"), "priority": "high"}
]


def print_tracking_info(planner, step_name):
    """Print tracking information at each step"""
    print("\n" + "=" * 30)
    print(f"TRACKING INFO: {step_name}")
    print("=" * 30)

    # Get planning status
    status = planner.get_planning_status()

    # Print current task and depth
    print(f"Current task: {status['current_task']}")
    print(f"Planning depth: {status['planning_depth']}")

    # Print task hierarchy
    print("\nTask Hierarchy:")
    print(planner.get_task_hierarchy_string())

    # Print execution trace
    print("\nExecution History:")
    print(planner.get_execution_trace())

    # Print method trace
    print("\nMethod Applications:")
    print(planner.get_current_method_trace())

    # Print operator trace
    print("\nOperator Applications:")
    print(planner.get_current_operator_trace())

    # Print backtrack points if any
    backtrack_points = planner.get_backtrack_points()
    if backtrack_points:
        print("\nBacktrack Points:")
        for point in backtrack_points:
            print(f"  {point['index']}: {point['task']} ({point['node_type']})")

    print("=" * 30 + "\n")


def main():
    # Create the agent
    agent = RobotAgent()
    agent.env.state = initial_state.copy()
    agent.env._update_internal_state()

    # Create the planner with the agent
    planner = NodePlanner(tasks=tasks, domain=domain, agent=agent)
    planner.update_state(initial_state)

    # Display initial state
    print("Initial state:")
    agent.env.render()

    # Add a debug log
    planner.debug_log("Starting planning process")

    print("\nStep-by-step planning with tracking:")
    print("-----------------------------------")

    # Step 1: Get the next decomposition (deliver_item)
    decomposition = planner.get_next_decomposition()
    task_node = decomposition['task_node']
    options = decomposition['options']

    print(f"Task: {task_node.content.name}")
    print(f"Found {len(options)} applicable methods/operators")

    # Print tracking info after first decomposition
    print_tracking_info(planner, "After first decomposition")

    # Step 2: Apply the deliver_item method
    result = planner.apply(task_node.id, 0)
    print(f"Applied method: {result['name']}")

    # Print tracking info after method application
    print_tracking_info(planner, "After deliver_item method")

    # Step 3: Apply the get_item method
    next_task_id = result['next_task_id']
    result = planner.apply(next_task_id, 0)
    print(f"Applied method: {result['name']}")

    # Print tracking info after method application
    print_tracking_info(planner, "After get_item method")

    # Step 4: Apply the move operator
    next_task_id = result['next_task_id']
    result = planner.apply(next_task_id, 0)
    print(f"Applied operator: {result['name']}")

    # Print tracking info after operator execution
    print_tracking_info(planner, "After move operator")

    # Step 5: Apply the pickup operator
    next_task_id = result['next_task_id']
    result = planner.apply(next_task_id, 0)
    print(f"Applied operator: {result['name']}")

    # Print tracking info after operator execution
    print_tracking_info(planner, "After pickup operator")

    # Step 6: Apply the move operator
    next_task_id = result['next_task_id']
    result = planner.apply(next_task_id, 0)
    print(f"Applied operator: {result['name']}")

    # Print tracking info after operator execution
    print_tracking_info(planner, "After second move operator")

    # Step 7: Apply the drop operator
    next_task_id = result['next_task_id']
    result = planner.apply(next_task_id, 0)
    print(f"Applied operator: {result['name']}")

    # Print tracking info after operator execution
    print_tracking_info(planner, "After drop operator")

    # Print final plan
    print("\nFinal plan:")
    for i, action in enumerate(planner.get_current_plan()):
        print(f"{i + 1}. {action.name} - {action.type}")


if __name__ == "__main__":
    main()

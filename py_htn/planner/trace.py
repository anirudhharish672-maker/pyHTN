from copy import deepcopy
import time


class TraceEntry:
    """Base class for all trace entries."""

    def __init__(self, entry_type, description):
        self.entry_type = entry_type
        self.description = description
        self.timestamp = time.time()


class MethodEntry(TraceEntry):
    """Records a method selection."""

    def __init__(self, task, method, reason):
        super().__init__("method", f"Applied method {method.name} to task {task.name}")
        self.task = task
        self.method = method
        self.reason = reason


class OperatorEntry(TraceEntry):
    """Records an operator execution."""

    def __init__(self, task, operator, success, state_before, state_after=None):
        status = "successfully" if success else "unsuccessfully"
        super().__init__("operator", f"Executed operator {operator.name} {status} for task {task.name}")
        self.task = task
        self.operator = operator
        self.success = success
        self.state_before = state_before
        self.state_after = state_after


class TaskEntry(TraceEntry):
    """Records a task decomposition."""

    def __init__(self, task, parent_task=None, parent_method=None):
        if parent_task:
            desc = f"Decomposing task {task.name} as subtask of {parent_task.name}"
        else:
            desc = f"Decomposing root task {task.name}"
        super().__init__("task", desc)
        self.task = task
        self.parent_task = parent_task
        self.parent_method = parent_method


class BacktrackEntry(TraceEntry):
    """Records a backtrack operation."""

    def __init__(self, from_task, to_task, reason):
        if to_task is None:
            desc = f"Backtracking from {from_task.name}. No parent tasks: {reason}"
        else:
            desc = f"Backtracking from {from_task.name} to {to_task.name}: {reason}"
        super().__init__("backtrack", desc)
        self.from_task = from_task
        self.to_task = to_task
        self.reason = reason


class Trace:
    """
    Maintains a comprehensive trace of the planning process including method decisions,
    operator executions, task decompositions, and backtracking operations.
    """

    def __init__(self):
        self.entries = []
        self.sequence_counter = 0

    def add_method(self, task, method, reason="Method applicable"):
        """Record a method selection."""
        entry = MethodEntry(task, method, reason)
        self.entries.append(entry)
        return entry

    def add_operator(self, task, operator, success, state_before, state_after=None):
        """Record an operator execution."""
        entry = OperatorEntry(task, operator, success, state_before, state_after)
        self.entries.append(entry)
        self.sequence_counter += 1 if success else 0
        return entry

    def add_task(self, task, parent_task=None, parent_method=None):
        """Record a task decomposition."""
        entry = TaskEntry(task, parent_task, parent_method)
        self.entries.append(entry)
        return entry

    def add_backtrack(self, from_task, to_task, reason):
        """Record a backtrack operation."""
        entry = BacktrackEntry(from_task, to_task, reason)
        self.entries.append(entry)
        return entry

    def get_current_plan(self):
        """
        Get the current plan consisting of successful operator executions.
        Returns a list of operators in the order they were executed.
        """
        plan = []
        for entry in self.entries:
            if entry.entry_type == "operator" and entry.success:
                plan.append(entry.operator)
        return plan

    def get_current_trace(self, include_states=False):
        """
        Get the full trace with all entries.
        If include_states is False, state information is omitted to reduce verbosity.
        """
        if include_states:
            return self.entries

        # Create a copy without state information
        simplified_trace = []
        for entry in self.entries:
            entry_copy = deepcopy(entry)
            if hasattr(entry_copy, 'state_before'):
                entry_copy.state_before = "..."
            if hasattr(entry_copy, 'state_after'):
                entry_copy.state_after = "..."
            simplified_trace.append(entry_copy)

        return simplified_trace

    def print_plan(self):
        """
        Print the current plan.
        """
        plan = self.get_current_plan()

        print("\n┌─────────────────────────────────────────────────┐")
        print("│                 CURRENT PLAN                    │")
        print("└─────────────────────────────────────────────────┘")

        if not plan:
            print("No actions in plan yet.")
            return

        for i, operator in enumerate(plan):
            print(f"Step {i + 1:02d}: {operator.name}({', '.join(str(arg) for arg in operator.args)})")

        print(f"\nTotal actions: {len(plan)}")
        print("────────────────────────────────────────────────────")

    def print_trace(self, include_states=False, max_entries=None):
        """
        Prints the full trace of the plan.
        :param include_states: Whether to include state information
        :param max_entries: Maximum number of entries to print (None for all)
        :return:
        """
        trace = self.get_current_trace(include_states)
        if max_entries:
            trace = trace[-max_entries:]

        print("\n┌─────────────────────────────────────────────────┐")
        print("│                PLANNING TRACE                   │")
        print("└─────────────────────────────────────────────────┘")

        if not trace:
            print("No trace entries yet.")
            return

        # Define colors for different entry types (ANSI escape codes)
        colors = {
            "task": "\033[94m",  # Blue
            "method": "\033[92m",  # Green
            "operator": "\033[93m",  # Yellow
            "backtrack": "\033[91m"  # Red
        }
        reset_color = "\033[0m"

        for i, entry in enumerate(trace):
            # Format timestamp
            timestamp = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))

            # Start with line number and timestamp
            line = f"{i + 1:03d} [{timestamp}] "

            # Add colored type indicator
            type_color = colors.get(entry.entry_type, "")
            type_str = f"{type_color}[{entry.entry_type.upper()}]{reset_color}"
            line += f"{type_str:12} "

            # Add description
            line += entry.description

            # Print line
            print(line)

            # Add additional info based on entry type
            if entry.entry_type == "method":
                print(f"     Reason: {entry.reason}")

            elif entry.entry_type == "operator" and include_states:
                print(f"     Success: {entry.success}")
                if include_states:
                    print(f"     State before: {entry.state_before}")
                    if entry.state_after:
                        print(f"     State after: {entry.state_after}")

            # Add separator between entries
            print("─" * 60)

        print(f"\nTotal entries: {len(trace)}")
        if max_entries and len(self.entries) > max_entries:
            print(f"Showing last {max_entries} of {len(self.entries)} entries.")
        print("────────────────────────────────────────────────────")

# Now I'll update the necessary functions in the HtnPlanner class
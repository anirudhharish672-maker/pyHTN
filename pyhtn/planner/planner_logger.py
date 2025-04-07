import inspect
import os
import logging
from time import strftime

class PlannerLogger:
    """
    A logger for the HTN planner that tracks and logs all planner actions.
    
    :param log_file: Path to the log file
    :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param console_output: Whether to also print logs to console
    """
    
    def __init__(self, log_file: str, log_level: int = logging.INFO, console_output: bool = False):
        """
        Initialize the planner logger.
        
        :param log_file: Path to the log file
        :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        :param console_output: Whether to also print logs to console
        """
        self.logger = logging.getLogger('htn_planner')
        self.logger.setLevel(log_level)
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter
        timestamp = strftime('%Y-%m-%d %H:%M:%S')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        
        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.info(f"HTN Planner Logger initialized at {timestamp}")
    
    def debug(self, message: str):
        """
        Log a debug message.
        
        :param message: Message to log
        """
        self.logger.debug(message)
    
    def info(self, message: str):
        """
        Log an info message.
        
        :param message: Message to log
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """
        Log a warning message.
        
        :param message: Message to log
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """
        Log an error message.
        
        :param message: Message to log
        """
        self.logger.error(message)
    
    def critical(self, message: str):
        """
        Log a critical message.
        
        :param message: Message to log
        """
        self.logger.critical(message)
    
    def log_task(self, task_node):
        """
        Log information about a task node.
        
        :param task_node: Task node to log
        """
        task_name = task_node.content.name if hasattr(task_node.content, 'name') else str(task_node.content)
        task_args = task_node.content.args if hasattr(task_node.content, 'args') else []
        task_id = task_node.id
        task_status = task_node.status
        
        self.info(f"Task: {task_name} (args: {task_args}, id: {task_id}, status: {task_status})")

    def log_function(self):
        # Get the frame of the calling function
        frame = inspect.currentframe().f_back

        # Get function name
        function_name = frame.f_code.co_name

        # Get argument names and values
        args, _, _, local_vars = inspect.getargvalues(frame)
        args_list = [f"{arg}={repr(local_vars[arg])}" for arg in args]
        args_str = ", ".join(args_list)

        # Print function name with arguments
        self.info(f"Entering {function_name}({args_str})")
    
    def log_method_application(self, task_node, method_node, subtasks):
        """
        Log a method application.
        
        :param task_node: Task node the method is applied to
        :param method_node: Method node being applied
        :param subtasks: Resulting subtasks
        """
        task_name = task_node.content.name if hasattr(task_node.content, 'name') else str(task_node.content)
        method_name = method_node.content.name if hasattr(method_node.content, 'name') else str(method_node.content)
        
        self.info(f"Applying Method: {method_name} to task {task_name}")
        self.debug(f"  Resulting in subtasks: {subtasks}")
    
    def log_operator_application(self, task_node, operator_node, effects):
        """
        Log an operator application.
        
        :param task_node: Task node the operator is applied to
        :param operator_node: Operator node being applied
        :param effects: Effects of the operator
        """
        task_name = task_node.content.name if hasattr(task_node.content, 'name') else str(task_node.content)
        operator_name = operator_node.content.name if hasattr(operator_node.content, 'name') else str(operator_node.content)
        
        self.info(f"Applying Operator: {operator_name} to task {task_name}")
        self.debug(f"  Effects: {effects}")
    
    def log_state_update(self, state):
        """
        Log a state update.
        
        :param state: The new state
        """
        self.debug(f"State updated: {state}")
    
    def log_backtrack(self, from_task, to_task):
        """
        Log a backtracking operation.
        
        :param from_task: Task being backtracked from
        :param to_task: Task being backtracked to
        """
        self.info(f"Backtracking from {from_task} to {to_task}")
    
    def log_plan_step(self, plan_action):
        """
        Log a plan step.
        
        :param plan_action: Plan action being added
        """
        self.info(f"Plan Step {plan_action.id}: {plan_action.type} - {plan_action.name}")
    
    def log_plan_complete(self, plan):
        """
        Log a completed plan.
        
        :param plan: The completed plan
        """
        self.info(f"Plan completed with {len(plan)} steps")
        for idx, action in enumerate(plan):
            self.info(f"  Step {idx+1}: {action.type} - {action.name}")

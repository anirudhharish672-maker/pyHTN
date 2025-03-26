class PlannerException(Exception):
    """
    A generic exception for miscellaneous errors in the planning process.

    :param message: Description of the error

    :attribute message: Description of the error
    """

    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class StopException(PlannerException):
    """
    Exception raised to indicate normal task completion or a stopping condition in planning.

    :param plan: The plan that was being executed when the exception occurred
    :param message: Explanation of why planning stopped

    :attribute plan: The plan that was being executed
    :attribute message: Explanation of why planning stopped
    """

    def __init__(self, plan=None, message="Task Completed"):
        self.plan = plan
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.plan} -> {self.message}'


class FailedPlanException(PlannerException):
    """
    Exception raised when plan execution fails due to unsatisfiable conditions or constraints.

    :param plan: The plan that failed during execution
    :param message: Explanation of why the plan failed

    :attribute plan: The plan that failed
    :attribute message: Explanation of why the plan failed
    """

    def __init__(self, plan=None, message="The plan execution failed"):
        self.plan = plan
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.plan} -> {self.message}'



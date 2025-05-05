from abc import abstractmethod
from copy import deepcopy
from itertools import chain
import logging
import time
from dataclasses import dataclass

from pyhtn.common.imports.typing import *
from pyhtn.htn import Task, Method, Operator, TaskEx, MethodEx, OperatorEx, ExStatus

# from pyhtn.domain.method import GroundedMethod
# from pyhtn.domain.method import Method
# from pyhtn.domain.operators import GroundedOperator
# from pyhtn.domain.operators import Operator
# from pyhtn.domain.task import TaskEx
# from pyhtn.domain.task import Task
from pyhtn.exceptions import FailedPlanException, StopException
from pyhtn.validation import validate_domain, validate_tasks
from pyhtn.planner.planner_logger import PlannerLogger
from pyhtn.planner.trace import Trace, TraceKind


@dataclass(eq=True, frozen=False)
class FrameContext:
    """
    Keeps track of the current position within an attempt
    to carry out a task exection
    """

    # The task execution being handled by this frame
    task_exec : TaskEx
    # Possible method executions for current task execution.
    #  If one fails we can backtrack and try another
    possible_method_execs : List[MethodEx]
    # Index of current MethodEx
    method_exec_index : int

    # MethodEx inidices already visited. Keep this list in case 
    #  for some reason we don't go through them in order
    visted_method_exec_indices : List[int]

    # Within the current MethodEx the index of the active subtask_exec
    subtask_exec_index : int
    # MethodEx inidices already visited. Keep this list in case 
    #  for some reason we don't go through them in order
    visted_subtask_exec_indices : List[int]

    def _next_method_exec_index(self):
        # TODO : Might consider policies other than trying MethodExs sequentially  
        nxt = self.method_exec_index + 1 
        if(nxt >= len(self.possible_method_execs)):
            return None
        return nxt

    def _next_subtask_exec_index(self):
        # TODO : Might consider policies other than trying MethodExs sequentially  
        nxt = self.subtask_exec_index + 1 
        subtask_execs = self.current_method_exec.subtask_execs
        if(nxt >= len(subtask_execs)):
            return None
        return nxt

    def next_method_frame(self):
        ind = self._next_method_exec_index()

        if(ind is None):
            return None

        return FrameContext(
            task_exec =                 self.task_exec,
            possible_method_execs =     self.possible_method_execs,
            method_exec_index =         ind, 
            visted_method_exec_indices = (
                self.visted_method_exec_indices + [self.method_exec_index],
            ),
            subtask_exec_index =         0,
            visted_subtask_exec_indices = [],
        )

    def next_subtask_frame(self):
        ind = self._next_subtask_exec_index()
        if(ind is None):
            return None

        return FrameContext(
            task_exec =                  self.task_exec,
            possible_method_execs =      self.possible_method_execs,
            method_exec_index =          self.method_exec_index, 
            visted_method_exec_indices = self.visted_method_exec_indices,
            subtask_exec_index =         ind,
            visted_subtask_exec_indices = (
                self.visted_subtask_exec_indices + [self.subtask_exec_index]
            ),
        )

    @classmethod
    def new_frame(self, task_exec, method_execs, 
                    method_index=0,
                    subtask_index=0):
        return FrameContext(
            task_exec =                  task_exec,
            possible_method_execs =      method_execs,
            method_exec_index =          method_index, 
            visted_method_exec_indices = [],
            subtask_exec_index =         subtask_index,
            visted_subtask_exec_indices = []
        )

    @property
    def current_task_exec(self):
        return self.task_exec

    @property
    def current_method_exec(self):
        method_execs = self.possible_method_execs
        meth_ind = self.method_exec_index
        if(method_execs and meth_ind is not None 
            and len(method_execs) > 0):

            return method_execs[meth_ind]
        else:
            return None

    @property
    def current_subtask_exec(self):
        if(not self.current_method_exec):
            return None
        subtask_execs = self.current_method_exec.subtask_execs
        ind = self.subtask_exec_index
        if(subtask_execs and ind is not None
           and len(subtask_execs) > 0):
            return subtask_execs[ind]
        else:
            return None


    @property
    def is_nomatch(self):
        return not self.possible_method_execs


class Cursor:
    def __init__(self):
        # Current task execution being processed
        self.trace = Trace()
        self.reset()

    def reset(self):
        self.current_frame = None
        self.stack = []

    def is_at_end(self):
        return self.current_frame is None


    @property
    def current_task_exec(self):
        return self.current_frame.current_task_exec

    @property
    def current_method_exec(self):
        return self.current_frame.current_method_exec

    @property
    def current_subtask_exec(self):
        return self.current_frame.current_subtask_exec

    def _pop_frame(self):
        if(len(self.stack) == 0):
            self.current_frame = None
            return None

        self.current_frame = self.stack[-1]
        self.stack.pop()
        return self.current_frame



    def advance_subtask(self, trace=None):
        ''' Try to move the cursor to the next position by: 
                1. Moving it laterally to the next subtask
                2. Popping up the stack if at last subtask in method

            This is the normal movement of the cursor after a successfully
            applying an operator.  
        ''' 
        while True:
            # Advance to next subtask exececution in current MethodEx
            curr_frame = self.current_frame
            curr_frame.current_subtask_exec.status = ExStatus.SUCCESS
            next_frame = self.current_frame.next_subtask_frame()
            if(next_frame is not None):
                if(trace):
                    trace.add(TraceKind.ADVANCE_SUBTASK, curr_frame, next_frame)
                break
            
            # If subtask sequence exhasted try popping frame from stack
            curr_frame.current_task_exec.status = ExStatus.SUCCESS
            curr_frame.current_method_exec.status = ExStatus.SUCCESS
            next_frame = self._pop_frame()
            if(next_frame is None):
                break
            if(trace):
                trace.add(TraceKind.POP_FRAME, curr_frame, next_frame)
                

        self.current_frame = next_frame

    def push_task_exec(self, task_exec, method_execs, method_ind=0, trace=None):
        ''' Recurse into a new frame from a task execution and
            its list of method executions
        ''' 
        # Push current frame 
        if(self.current_frame is not None):
            self.stack.append(self.current_frame)

        # Make a new frame pointing at the selected MethodEx
        curr_frame = self.current_frame        
        next_frame = FrameContext.new_frame(task_exec, method_execs, method_ind)
        next_frame.current_task_exec.status = ExStatus.IN_PROGRESS
        next_frame.current_method_exec.status = ExStatus.IN_PROGRESS
        if(trace):
            trace.add(TraceKind.SELECT_METHOD,       curr_frame, next_frame)
            trace.add(TraceKind.FIRST_SUBTASK, curr_frame, next_frame)
        self.current_frame = next_frame



    def backtrack(self, trace_kind=None, trace=None):
        ''' Pop frame off the stack and try the next method execution.
            Keep popping up stack until a method execution is found.
            This is the normal movement of the cursor after failing to 
            apply an operator as part of a subtask sequence.
        '''
        while True:
            curr_frame = self.current_frame
            curr_frame.current_task_exec.status = ExStatus.FAILED
            curr_frame.current_method_exec.status = ExStatus.FAILED
            curr_frame.current_subtask_exec.status = ExStatus.FAILED
            next_frame = self._pop_frame()

            if(trace):
                trace.add(trace_kind, curr_frame, next_frame)

            if not next_frame:
                return False

            
            # Try next possible MethodEx
            curr_frame = self.current_frame
            next_frame = self.current_frame.next_method_frame()
            if(next_frame is not None):
                if(trace):
                    trace.add(TraceKind.SELECT_METHOD,       curr_frame, next_frame)
                    trace.add(TraceKind.FIRST_SUBTASK, curr_frame, next_frame)
                break

            # If no more MethodEx then pop up again
            trace_kind = TraceKind.BACKTRACK_CHILD_CASCADE

        self.current_frame = next_frame
        return True

    def user_select_method_exec(self, method_ind=0, trace=None):
        # Make a new frame pointing at the selected MethodEx
        curr_frame = self.current_frame

        if(curr_frame.current_method_exec):
            curr_frame.current_method_exec.status = ExStatus.INITIALIZED

        next_frame = FrameContext.new_frame(
            curr_frame.task_exec,
            curr_frame.possible_method_execs,
            method_ind)

        next_frame.current_method_exec.status = ExStatus.IN_PROGRESS

        if(trace):
            trace.add(TraceKind.USER_SELECT_METHOD,  curr_frame, next_frame)
            trace.add(TraceKind.FIRST_SUBTASK, curr_frame, next_frame)

        self.current_frame = next_frame

    def push_nomatch_frame(self, task_exec, method_execs, trace=None):
        ''' Recurse into an nomatch frame this is a kind of 
            frame where method_execs is None or []. In normal planning
            this is an invalid frame. When using the planner 
            interactively (like in VAL) this is a valid state in which
            is used in 
        ''' 
        curr_frame = self.current_frame

        # Push current frame 
        if(self.current_frame is not None):
            self.stack.append(self.current_frame)

        next_frame = FrameContext.new_frame(
            task_exec,
            method_execs,
            None)

        if(trace):
            trace.add(TraceKind.ENTER_NOMATCH_FRAME, curr_frame, next_frame)

        self.current_frame = next_frame

    def add_method_exec(self, method_exec, task_exec=None, trace=None):
        # Resolve the frame associated with task_exec
        task_frame = None
        if(task_exec is None):
            task_frame = self.current_frame
        else:
            for frame in [self.current_frame, self.stack]:
                if(task_exec is frame.current_task_exec):
                    task_frame = frame
                    break

        if(task_frame is None):
            raise ValueError(f"No frame in plan stack associated with {task_exec}.")

        # Add the method_exec to the possibilities in the frame
        if(task_frame.possible_method_execs is None):
            task_frame.possible_method_execs = []
        task_frame.possible_method_execs.append(method_exec)

        if(trace):
            trace.add(TraceKind.USER_ADD_METHOD, method_exec, task_frame)


    def print(self):
        """
        Print the current state of the cursor in a readable format.
        """
        print("========== CURSOR STATE ==========")

        frame = self.current_frame
        if(self.is_at_end()):
            print("-- Cursor reached end of its root task -- ")
            return

        te = self.current_task_exec
        if(te):
            # TODO: status
            # print(f"Current TaskEx: {te} (Status: {te.status})")
            print(f"Current TaskEx: {te}")
        else:
            print(f"Current TaskEx: {te}")

        me = self.current_method_exec
        ste = self.current_subtask_exec
        st_ind = frame.subtask_exec_index
        if me:
            print(f"Current MethodEx: {me}")
            print(f"Current SubtaskEx: {ste} ({st_ind+1}/{len(me.subtask_execs)})")
        else:
            print("Current Method: None")
            print("Current Subtask Index: N/A")

        print(f"Possible MethodExs: {len(frame.possible_method_execs)}")
        print(f"Current Method Index: {frame.method_exec_index}")
        print(f"Stack Depth: {len(self.stack)}")

        if self.stack:
            print("-- Stack (most recent first) --")
            for i, stack_frame in enumerate(reversed(self.stack[-3:])):
                te = frame.current_task_exec
                me = frame.current_method_exec
                ste = frame.current_subtask_exec

                print(f"  {i}: {te}, '{me}', {ste}")

            if len(self.stack) > 3:
                print(f"  ... and {len(self.stack) - 3} more")

        print("==================================\n")


class TaskManager:
    def __init__(self, choice_criterion: str = 'random'):
        self.queue = []
        self.options = ()
        self.choice_criterion = choice_criterion

    def flatten(self):
        """
        Convert partially ordered task list into total ordered task list
        based on choice resolution criteria.
        """
        pass

    @abstractmethod
    def get_next_task(self):
        if not self.options:
            if not self.queue:
                return None
            self.options = tuple(self._get_next_tasks(self.queue))
        if self.options:
            if self.choice_criterion == 'random':
                pass


    def _get_next_tasks(self, tasks: Union[List, Tuple]) -> Union[List, Tuple]:
        """
        Returns list/tuple of tasks which no other task is constrained to precede.
        """
        if isinstance(tasks, list) and not isinstance(tasks[0], (list, tuple)):
            return list([tasks.pop()])
        elif isinstance(tasks, list):
            return self._get_next_tasks(tasks[0])
        elif isinstance(tasks, tuple):
            return tuple(
                chain.from_iterable(self._get_next_tasks(t) if isinstance(t, (list, tuple)) else (t,) for t in tasks))



class RootTaskQueue(TaskManager):
    def __init__(self, tasks: List[Union[Task,dict]],
                       choice_criterion: str = 'random',
                       enable_logging=False):
        """
        :param task_specs: A list of root tasks or task specifications with priorities  
        :param choice_criterion: How partially ordered tasks should be selected. One of [random, ordered, cost]
        """
        super().__init__(choice_criterion)
        self.enable_logging = enable_logging

        for task in tasks:
            self.add(task)
        

    def add(self, task: Union[dict, tuple, Task]):
        if(isinstance(task, dict)):
            task = Task(**task)
        elif(isinstance(task, tuple)):
            task = Task(*task)
            
        if task.priority == 'first' or self.is_empty():
            self.queue.insert(0, task)
            if self.enable_logging:
                self.logger.info(f"Added task {task} as first task")
        else:
            priority_map = {'first': 0, 'high': 1, 'medium': 2, 'low': 3}
            task_priority = priority_map.get(task.priority)

            for i, existing_task in enumerate(self.queue):
                existing_priority = existing_task.tasks.priority
                if isinstance(existing_priority, str):
                    existing_priority = priority_map.get(existing_priority, 3)

                if task_priority < existing_priority:
                    pos = i
                    self.queue.insert(i, task)
                else:
                    pos = i + 1
                self.queue.insert(pos, task)
                if self.enable_logging:
                    self.logger.info(f"Added task {task} at position {pos} with priority {task.priority}")
                return

            self.queue.append(task)
            if self.enable_logging:
                self.logger.info(f"Added task {task} at the end with priority {task.priority}")

    def get_next_task(self) -> Task:
        if(self.is_empty()):
            return None
        return self.queue.pop(0)

    def is_empty(self):
        return len(self.queue) == 0

    def clear(self):
        self.queue =[]

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        return iter(self.queue)



class HtnPlanner2:
    """
    This planner implements HTN (Hierarchical Task Network) planning using a graph of
    task, method, and operator nodes.
    """

    def __init__(self,
                 domain: Dict[str, List[Method]],
                 tasks: List[dict] = [],
                 env: Any = None,
                 validate_input: bool = False,
                 repeat_wait_time: float = 0.1,
                 order_by_cost: bool = False,
                 enable_logging: bool = False,
                 log_dir: str = None,
                 log_level: int = logging.INFO,
                 console_output: bool = False):
        """
        Initialize the HTN planner.
        :param tasks:
        :param domain:
        :param env:
        :param validate_input:
        :param repeat_wait_time:
        :param order_by_cost:
        :param enable_logging:
        :param log_dir:
        :param log_level:
        :param console_output:
        """
        # Validate inputs if requested
        if validate_input:
            validate_domain(domain)
            validate_tasks(tasks)

        # Initialize instance variables
        self.env = env
        self.state = self.env.get_state()

        # Replace plan_list with trace
        self.trace = Trace()

        # Planning options
        self.validate = validate_input
        self.repeat_wait_time = repeat_wait_time
        self.order_by_cost = order_by_cost
        self.enable_logging = enable_logging

        print("enable_logging", enable_logging)
        
        # Setup logger
        if self.enable_logging:
            log_file = log_dir if log_dir is not None else 'planner_logs/planner_log.log'
            self.logger = PlannerLogger(log_file, log_level, console_output)
            self.logger.info("HTN Planner initialized")
        else:
            self.logger = None

        if self.enable_logging:
            self.logger.info(f"Added {len(tasks)} root tasks")
        self.root_task_queue = RootTaskQueue(tasks)

        self.planning_start_time = None

        # Set domain network
        self.domain_network = domain

        # Create cursor to track planning state
        self.cursor = Cursor()

        if self.enable_logging:
            self.logger.info(f"Domain network built with {len(domain)} task types")
            self.logger.info(f"Domain network structure: {list(self.domain_network.keys())}")


    # def get_current_trace(self, include_states: bool = False):
    #     """

    #     :param include_states: Whether to include state information in the output
    #     :return:
    #     """
    #     """Gets the current trace."""
    #     if self.enable_logging:
    #         self.logger.log_function()
    #     return self.trace.get_current_trace(include_states)

    def is_exhausted(self):
        return self.root_task_queue.is_empty() and self.cursor.is_at_end()

    # --------------------------
    # : plan()

    def _plan_start(self):
        self.planning_start_time = time.time()
        if self.enable_logging:
            self.logger.info("Starting automated planning")

        if self.env and hasattr(self.env, 'get_state'):
            self.state = self.env.get_state()

        if(self.is_exhausted()):
            raise StopException("There are no tasks to plan for")

    def _next_task_exec(self):
        # If cursor is exhausted, pop off new root task, and make it into a TaskEx.
        if(self.cursor.is_at_end()):
            if(self.root_task_queue.is_empty()):
                self.trace.add(TraceKind.ROOT_TASKS_EXHAUSTED)
                return None

            root_task = self.root_task_queue.get_next_task()
            task_exec = root_task.as_task_exec(self.state)
            self.trace.add(TraceKind.NEW_ROOT_TASK, task_exec)

        # Otherwise use the TaskEx that the cursor is pointing to
        else:
            task_exec = self.cursor.current_subtask_exec

        return task_exec

    def _handle_task_exec(self, task_exec, push_nomatch_frames=False):
        ''' Handle the execution of a TaskEx instance
            :return: A TaskKind Enum signaling how the execution was carried out
        '''

        # Apply pattern matching to find child MethodExs or OperatorEx
        #   of the current TaskEx.
        print("get_child_executions", task_exec)
        child_execs = task_exec.get_child_executions(
            self.domain_network, self.state
        )

        # Backtrack or push a nomatch frame if matching fails for any reason.
        if(child_execs is None or len(child_execs) == 0):
            if(push_nomatch_frames):
                print("NO MATCH FRAME", child_execs)
                self.cursor.push_nomatch_frame(task_exec, child_execs, trace=self.trace)
                return TraceKind.FIRST_SUBTASK
            else:
                
                backtrack_kind = (
                    TraceKind.BACKTRACK_NO_CHILDREN,
                    TraceKind.BACKTRACK_NO_MATCH
                )[child_execs is not None]

                self.cursor.backtrack(backtrack_kind, trace=self.trace)
                return backtrack_kind

        # Primitive Task Case: execute Operator
        if(isinstance(child_execs[0], OperatorEx)):
            operator_exec = child_execs[0]
            success = self._apply_operator_execution(operator_exec)
            if(success):
                self.trace.add(TraceKind.APPLY_OPERATOR, operator_exec)
                self.cursor.advance_subtask(trace=self.trace)
                return TraceKind.ADVANCE_SUBTASK
            else:
                self.cursor.backtrack(
                    TraceKind.BACKTRACK_OPERATOR_FAIL,
                    trace=self.trace
                )
                return TraceKind.BACKTRACK_OPERATOR_FAIL
        # Higher-Oder Task Case: Push a new frame with options for 
        #    executing the task (will be handled in next loop).
        else:
            self.cursor.push_task_exec(task_exec, child_execs, trace=self.trace)
            task_exec.status = ExStatus.IN_PROGRESS
            return TraceKind.FIRST_SUBTASK


    def plan(self, stop_kinds=[], push_nomatch_frame=False) -> List:
        """
        Execute the planning process.
        :return: The complete plan.
        """
        self._plan_start()

        # Main Plan Loop:
        #  Each loop handles one TaskEx instance and makes one move of the Cursor.
        #   (the Cursor holds the current plan state and stack).
        #  Every Cursor movement is gaurenteed to either:
        #   1. Enter a valid frame w/ a parent TaskEx, MethodEx, and current sub-TaskEx.
        #   2. Exhaust the cursor (the current TaskEx will come from the root_task_queue)
        while True:
            task_exec = self._next_task_exec()

            if(task_exec is None):
                break

            # Handle the current task execution
            trace_kind = self._handle_task_exec(task_exec, push_nomatch_frame)
            if(trace_kind in stop_kinds):
                return self.trace

        return self.trace
            

    # --------------------------
    # : printing methods

    def print_current_plan(self):
        """Prints the current plan."""
        if self.enable_logging:
            self.logger.log_function()
        return self.trace.print_plan()

    def print_current_trace(self, include_states: bool = False) -> None:
        """
        Prints the current trace.
        :param include_states: Whether to include state information in the output
        :return:
        """
        if self.enable_logging:
            self.logger.log_function()
        return self.trace.print_trace(include_states)

    def print_network(self) -> None:
        """
        Visualize the planner network.
        :return: None
        """
        if self.enable_logging:
            self.logger.log_function()
        strings = []
        tab = '  '
        header = "PLANNER NETWORK"
        border = '#' * (len(header) + 4)
        print(border)
        print('# ' + header + ' #')
        print(border + '\n')

        for task_key, methods_or_op in self.domain_network.items():
            strings.append(0 * tab + f'Task({task_key})')
            if(not isinstance(methods_or_op, list)):
                methods_or_op = [methods_or_op]

            for method_or_op in methods_or_op:
                if(isinstance(method_or_op, Method)):
                    method = method_or_op
                    pos_args = [method.name, *method.args]
                    strings.append(1 * tab + f"Method({', '.join([repr(x) for x in pos_args])}")

                    for subtask in method.subtasks:
                        strings.append(2 * tab + f'{subtask})')

                elif(isinstance(method_or_op, Operator)):                
                    strings.append(1 * tab + f'{method_or_op}')

                
        for s in strings:
            print(s)
        print('\n')

    def print_planner_state(self):
        if self.enable_logging:
            self.logger.log_function()
        print("===== PLANNER STATE =====")
        print(f"Root tasks: {len(self.root_task_execs)}")
        if not self.root_task_queue.is_empty():
            for i, task in enumerate(self.root_task_queue.queue):
                print(f"  Task {i}: {task.name} (status: {task.status})")

        print(f"Current task: {self.cursor.current_task.name if self.cursor.current_task else None}")
        print(f"Current method: {self.cursor.current_method.name if self.cursor.current_method else None}")
        print(f"Current subtask index: {self.cursor.current_subtask_index}")
        print(f"Available methods: {len(self.cursor.available_method_execs)}")
        print(f"Current method index: {self.cursor.current_method_index}")
        print(f"Stack depth: {len(self.cursor.stack)}")

        print(f"Trace entries: {len(self.trace.entries)}")
        for i, entry in enumerate(self.trace.entries[-5:]):  # Show last 5 entries
            print(f"  Entry {i}: type={entry.entry_type}")

        print("=========================")


    # ----------------------
    # : Adding Removing Root Tasks and Methods

    # def add_method(self, task_name: str, task_args: tuple['V'], preconditions: 'Fact', subtasks: List[Any]):
    #     if self.enable_logging:
    #         self.logger.log_function()
    #     new_method = Method(name=task_name, args=task_args, preconditions=preconditions, subtasks=subtasks)
    #     task_id = f"{task_name}/{len(task_args)}"
    #     self.domain_network[task_id].append(new_method)
    #     return new_method

    def add_method(self, method: Method):   
        task_id = str(method.name)
        if task_id not in self.domain_network:
            self.domain_network[task_id] = []
        self.domain_network[task_id].append(method)


    def add_tasks(self, tasks: Union[
                         Union[dict, tuple, Task],
                         Sequence[Union[dict, tuple, Task]]
                        ]) -> None:
        """
        Add tasks to the planner's root task queue.
        :param tasks: A list of task specifications.
        :return: None
        """
        if self.enable_logging:
            self.logger.log_function()

        if not isinstance(tasks, list):
            tasks = [tasks]

        for task in tasks:
            self.root_task_queue.add(task)
        # self._add_tasks(tasks)

    def clear_tasks(self):
        """Clear all tasks."""
        if self.enable_logging:
            self.logger.log_function()
        self.root_task_queue.clear()
        self.cursor.reset()

    def get_current_root(self):
        """Gets the current root."""
        if self.enable_logging:
            self.logger.log_function()

        if(len(self.cursor.stack) > 0):
            return self.cursor.stack[0].task_exec
        else:
            return self.cursor.current_frame.task_exec

    def get_current_plan(self):
        """Gets the current plan."""
        if self.enable_logging:
            self.logger.log_function()
        return self.trace.get_current_plan()

    def remove_task(self, task: str):
        """
        Remove the next occurrence of a task. If the provided task is the current task,
        it will be removed and the planner will automatically continue to the next text.
        """
        if self.enable_logging:
            self.logger.log_function()
        pass

    def remove_tasks(self, tasks: List[str]):
        """
        Remove all occurrences of the tasks in the given list.
        """
        if self.enable_logging:
            self.logger.log_function()
        pass

    def reset(self):
        if self.enable_logging:
            self.logger.log_function()
        self.trace = Trace()
        self.cursor = Cursor()
        self.root_task_execs = []

# --------------------------------------------
# : Interactive Interface

    def plan_to_next_decomposition(self, push_nomatch_frame=True):
        """
        Steps to the next applicable method for a task.
        :param all_methods: Whether to return all methods or one method at a time
        :return: The current task and either next method or all methods
        """
        return self.plan(stop_kinds=[TraceKind.FIRST_SUBTASK], push_nomatch_frame=push_nomatch_frame)

    def get_next_method_execs(self):
        frame = self.cursor.current_frame
        method_execs = frame.possible_method_execs
        return frame.task_exec, method_execs

    def stage_method_exec(self, method_exec):
        task_exec, method_execs = self.get_next_method_execs()
        inds = [i for i, x in enumerate(method_execs) if x == method_exec]

        if(len(inds) == 0):
            raise ValueError(f"MethodEx stage {method_exec} fail." +
                "Not a possible MethodEx for this state")

        self.cursor.user_select_method_exec(inds[0], self.trace)
    
    def add_method_exec(self, method_exec, task_exec=None):
        self.cursor.add_method_exec(method_exec, task_exec, self.trace)

        # if self.enable_logging:
        #     self.logger.log_function()
        # # Clear trace
        # # self.trace = Trace()
        # if not self.cursor.current_task or self.cursor.current_task.status == 'succeeded':
        #     if not self.root_task_execs:
        #         if self.enable_logging:
        #             self.logger.error("No tasks to plan for")
        #         raise StopException("There are no tasks to plan for. You must add tasks to the planner first"
        #                             " using add_tasks().")
        #     if not self._set_cursor_to_new_task():
        #         return self.cursor.current_task, None

        # if self.enable_logging:
        #     self.logger.info(f"Getting the next applicable method for task ({self.cursor.current_task.name})")

        # # No more methods
        # if self.cursor.current_method_index >= len(self.cursor.available_method_execs):
        #     if self.enable_logging:
        #         self.logger.info(f"All methods for task ({self.cursor.current_task.name}) have been returned."
        #                           f" No remaining methods.")
        #     return self.cursor.current_task, None

        # # Return all methods
        # if all_methods:
        #     # Set method index to end of list so that if called again, there is nothing to return
        #     # self.cursor.current_method_index = len(self.cursor.available_method_execs)
        #     self.cursor.current_method_index = len(self.domain_network[self.cursor.current_task.id])
        #     # return self.cursor.current_task, self.cursor.available_method_execs
        #     return self.cursor.current_task, self.domain_network[self.cursor.current_task.id]


        # # Get next method
        # method = self.cursor.available_method_execs[self.cursor.current_method_index]
        # self.cursor.current_method = method


        # # Add to trace
        # self.trace.add_method(
        #     self.cursor.current_task,
        #     method,
        #     f"Method #{self.cursor.current_method_index + 1} shown during interactive stepping"
        # )

        # # Advance to next method for next call
        # self.cursor.current_method_index += 1

        # if self.enable_logging:
        #     self.logger.info(f"Providing method: {method.name} for task {self.cursor.current_task.name}")

        # return self.cursor.current_task, [method]

    # def apply_method_application(self, task: TaskEx, method_to_apply: Any):
    #     """
    #     Steps to the next subtask of a method.
    #     :param task: The current task
    #     :param method_to_apply: The method to apply to the task
    #     :return: The plan after applying the method
    #     """
    #     if self.enable_logging:
    #         self.logger.log_function()
    #     """
    #     if not self.cursor.current_task:
    #         self.cursor.current_task = task
    #     self.cursor.available_method_execs.append(method_to_apply.method)
    #     self.cursor.current_method_index = 0
    #     self.cursor.current_method = method_to_apply.method
    #     self.cursor.current_subtask_index = 0
    #     """
    #     if not self.cursor.current_task:
    #         self.cursor.set_task(task)
    #     self.cursor.available_method_execs.append(method_to_apply.method)
    #     self.cursor.current_method = method_to_apply.method



    #     # Add to trace
    #     self.trace.add_method(
    #         self.cursor.current_task,
    #         method_to_apply.method,
    #         f"Applying subtasks of method ({method_to_apply.method.name})"
    #     )
    #     print(f"Method in apply method application: {method_to_apply.method}")
    #     # Set cursor to this method
    #     # self.cursor.available_method_execs.append(method)
    #     # self.cursor.current_method = method
    #     # self.cursor.current_subtask_index = 0
    #     # self.cursor.current_method_index = 0

    #     return self.plan(interactive=True)



    ####################
    # HELPER FUNCTIONS #
    ####################

    def _apply_operator_execution(self, operator_exec: OperatorEx) -> bool:
        """Execute an operator and update state."""

        # Execute in environment if available

        state_before = deepcopy(self.state)
        
        success = True
        if self.env:
            if self.enable_logging:
                self.logger.info(f"Executing operator in environment: {operator_exec}")

            # Try to apply the operator in the environment
            try:
                success = self.env.execute_action(
                    operator_exec.operator.name,
                    operator_exec.match
                )
            except Exception as e:
                success = False
        else:
            if self.enable_logging:
                self.logger.info(f"No environment, skipping: {operator_exec}")
        
        if(success):
            operator_exec.status = ExStatus.SUCCESS
            if self.enable_logging:
                self.logger.info(f"State updated after operator execution: {operator_exec}")
        else:
            operator_exec.status = ExStatus.FAILED
            if self.enable_logging:
                self.logger.info(f"Operator execution failed: {operator_exec}")

        # Update planner state with environment state after applying operator
        if hasattr(self.env, 'get_state'):
            self.state = self.env.get_state()
            state_after = deepcopy(self.state)
        else:
            state_after = state_before

        return success
            
                

                # # Add operator to trace
                # self.trace.add_operator(
                #     self.cursor.current_task,
                #     result,
                #     success,
                #     state_before=state_before,
                #     state_after=state_after
                # )

                # # Check execution result
                # if not success:
                #     if self.enable_logging:
                #         self.logger.error(f"Operator {result.name} failed in environment")
                #     self.cursor.current_task.status = 'failed'
                #     return False

            
                # self.cursor.current_task.status = 'failed'

                # # Add failed operator to trace
                # self.trace.add_operator(
                #     self.cursor.current_task,
                #     result,
                #     False,
                #     state_before=deepcopy(self.state),
                #     state_after=None
                # )

                # if self.enable_logging:
                #     self.logger.error(f"Failed to execute operator {result.name}: {str(e)}")

                
        # else:
        #     # No environment, just add to trace
        #     self.trace.add_operator(
        #         self.cursor.current_task,
        #         result,
        #         True,
        #         state_before=deepcopy(self.state),
        #         state_after=deepcopy(self.state)
        #     )

        # if self.enable_logging:
        #     self.logger.info(f"Executing operator {result.name} was successful")
        # return True

    # def _add_task_exec(self, task_exec: TaskEx) -> None:
    #     """Add a task to the root tasks queue based on priority."""
    #     if self.enable_logging:
    #         self.logger.log_function()
    #     if task_exec.task.priority == 'first' or not self.root_task_execs:
    #         self.root_task_execs.insert(0, task_exec)
    #         if self.enable_logging:
    #             self.logger.info(f"Added task {task.name} as first task")
    #     else:
    #         priority_map = {'first': 0, 'high': 1, 'medium': 2, 'low': 3}
    #         task_priority = priority_map.get(task_exec.task.priority)

    #         for i, existing_task in enumerate(self.root_task_execs):
    #             existing_priority = existing_task.tasks.priority
    #             if isinstance(existing_priority, str):
    #                 existing_priority = priority_map.get(existing_priority, 3)

    #             if task_priority < existing_priority:
    #                 pos = i
    #                 self.root_task_execs.insert(i, task_exec)
    #             else:
    #                 pos = i + 1
    #             self.root_task_execs.insert(pos, task_exec)
    #             if self.enable_logging:
    #                 self.logger.info(f"Added task {task_exec} at position {pos} with priority {task.priority}")
    #             return

    #         self.root_task_execs.append(task)
    #         if self.enable_logging:
    #             self.logger.info(f"Added task {task_exec} at the end with priority {task.priority}")

    # def _add_tasks(self, tasks: List[Union[str, Tuple, Dict]]) -> None:
    #     """
    #     Internal method for adding tasks to the planning queue.
    #     :param tasks: A list of task specifications.
    #     :return: None
    #     """
    #     if self.enable_logging:
    #         self.logger.log_function()
    #     if self.validate:
    #         validate_tasks(tasks)

    #     for t in tasks:
    #         task_name = t.get('name')
    #         args = t.get('arguments', ())
    #         priority = t.get('priority', 'low')
    #         repeat = t.get('repeat', False)

    #         task_exec = TaskEx(task_name, args=args, priority=priority, repeat=repeat)
    #         self._add_task_exec(task_exec)

    #         if self.enable_logging:
    #             self.logger.info(f"Added task: ({task_exec}) with priority ({priority}), repeat={repeat}")

    # def _backtrack(self) -> bool:
    #     """
    #     Attempt to backtrack to an alternative method.
    #     :return: Whether planner is able to backtrack to an alternative method.
    #     """
    #     if self.enable_logging:
    #         self.logger.log_function()
    #         self.logger.info(f"Attempting to backtrack")

    #     # Try to backtrack with the cursor
    #     from_task = self.cursor.current_task
    #     result = self.cursor.backtrack()

    #     if result:
    #         # Add backtrack entry to trace
    #         self.trace.add_backtrack(
    #             from_task,
    #             self.cursor.current_task,
    #             f"Trying alternative method {self.cursor.current_method.name}"
    #         )

    #         # Add method entry for the new method
    #         self.trace.add_method(
    #             self.cursor.current_task,
    #             self.cursor.current_method,
    #             "Method selected after backtracking"
    #         )

    #         if self.enable_logging:
    #             self.logger.info(f"Backtracked to task {self.cursor.current_task.name}, "
    #                              f"trying method {self.cursor.current_method.name}")
    #         return True
    #     else:
    #         if self.enable_logging:
    #             self.logger.warning("Backtracking failed, no alternative methods available")

    #         # Add backtrack entry to trace
    #         if self.cursor.stack:
    #             parent_task = self.cursor.stack[-1][0]
    #             self.trace.add_backtrack(
    #                 from_task,
    #                 parent_task,
    #                 "No alternative methods available"
    #             )
    #         else:
    #             self.trace.add_backtrack(
    #                 from_task,
    #                 None,
    #                 "No alternative methods available and no parent task"
    #             )

    #         return False

    # def _continue_method_execution(self) -> bool:
    #     """Continue executing subtasks of the current method."""
    #     if self.enable_logging:
    #         self.logger.log_function()
    #     if not self.cursor.current_method:
    #         return False

    #     subtasks = self.cursor.current_method.subtasks

    #     if self.enable_logging:
    #         self.logger.info(f"Continuing method {self.cursor.current_method.name} execution "
    #                          f"from subtask {self.cursor.current_subtask_index+1} of {len(subtasks)}")

    #     # Process each subtask starting from current index
    #     while self.cursor.current_subtask_index < len(subtasks):
    #         subtask = subtasks[self.cursor.current_subtask_index]

    #         if isinstance(subtask, OperatorEx):
    #             # Execute operator
    #             result = self._execute_operator(subtask)
    #             if not result:
    #                 return False
    #             # Move to next subtask
    #             self.cursor.current_subtask_index += 1


    #         elif isinstance(subtask, TaskEx):
    #             if self.enable_logging:
    #                 self.logger.info(f"Encountered subtask {subtask.name}")

    #             # Add task entry to trace
    #             self.trace.add_task(
    #                 subtask,
    #                 parent_task=self.cursor.current_task,
    #                 parent_method=self.cursor.current_method
    #             )

    #             # Save current context before moving to subtask
    #             self.cursor.push_context()
    #             # Change current task to subtask
    #             self.cursor.set_task(subtask)

    #             # Return true to indicate we've moved to a subtask
    #             # and should continue from there
    #             return True

    #     # All subtasks completed successfully
    #     self.cursor.current_task.status = 'succeeded'
    #     if self.enable_logging:
    #         self.logger.info(f"All subtasks completed for task {self.cursor.current_task.name}")

    #     return True

    

    def _get_method_executions(self) -> List[MethodEx]:
        """Get methods applicable to the current task."""
        if self.enable_logging:
            self.logger.log_function()
            self.logger.info(f"Locating applicable methods for task {self.cursor.current_task.name}")

        if not self.state:
            raise ValueError("State must be set before getting applicable methods")

        # Check if there are methods for this task type
        if self.cursor.current_task.id not in self.domain_network:
            if self.enable_logging:
                self.logger.warning(f"No methods found for task {self.cursor.current_task.id}")
            return []

        applicable_methods = []
        methods = self.domain_network[self.cursor.current_task.id]

        # Sort by cost if requested
        if self.order_by_cost:
            methods = sorted(methods, key=lambda m: m.cost)

        # Check each method
        visited = []
        for method in methods:
            result = method.applicable(self.cursor.current_task, self.state, str(self.trace.get_current_plan()), visited)
            if result:
                applicable_methods.append(result)
                if self.enable_logging:
                    self.logger.info(f"Method {method.name} is applicable to task {self.cursor.current_task.name}")

        if self.enable_logging:
            self.logger.info(
                f"Found {len(applicable_methods)} applicable methods for task {self.cursor.current_task.name}")

        return applicable_methods

    def _process_current_task(self) -> bool:
        """Process the current task by finding and applying methods."""
        if self.enable_logging:
            self.logger.log_function()
            self.logger.info(f"Processing task {self.cursor.current_task.name}")

        # Add task entry to trace
        self.trace.add_task(self.cursor.current_task)

        # If we already have a method, continue executing it
        if self.cursor.current_method:
            return self._continue_method_execution()

        # Get applicable methods for this task
        method_execs = self._get_method_executions()
        if not method_execs:
            if self.enable_logging:
                self.logger.warning(f"No applicable methods for task {self.cursor.current_task.name}")
            self.cursor.current_task.status = 'failed'
            return False

        # Store available methods for backtracking
        self.cursor.available_method_execs = method_execs
        self.cursor.current_method_index = 0
        self.cursor.current_method = methods[0]
        self.cursor.current_subtask_index = 0

        # Add method entry to trace
        self.trace.add_method(
            self.cursor.current_task,
            self.cursor.current_method,
            "Method selected as first applicable method"
        )

        if self.enable_logging:
            self.logger.info(
                f"Selected method {self.cursor.current_method.name} for task {self.cursor.current_task.name}")

        # Execute the selected method
        return self._continue_method_execution()

    def _set_cursor_to_new_task(self):
        if self.enable_logging:
            self.logger.log_function()
        next_task = self.root_task_execs.pop(0)
        if self.enable_logging:
            self.logger.info(f"Moving on to new task: ({next_task.name})")
        # Handle repeat tasks
        if next_task.repeat > 0:
            next_task.repeat -= 1
            self.root_task_execs.insert(0, next_task)
        self.cursor.set_task(next_task)
        # Get and store applicable methods
        method_execs = self._get_method_executions()
        if not method_execs:
            if self.enable_logging:
                self.logger.info(f"No applicable methods found for task: ({self.cursor.current_task.name})")
            return False

        self.cursor.available_method_execs = methods
        self.cursor.current_method_index = 0
        return True





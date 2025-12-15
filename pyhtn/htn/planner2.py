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

    # Child method or operator executions for current task execution.
    #  If one fails we can backtrack and try another
    child_execs : List[MethodEx]

    # Index of current MethodEx
    child_exec_index : int

    # MethodEx indices already visited. Keep this list in case 
    #  for some reason we don't go through them in order
    visited_child_exec_inds : List[int]

    # Within the current MethodEx the index of the active subtask_exec
    subtask_exec_index : int

    # The span of subtasks that this frame effectively covers 
    #   because they are part of an unordered group or are skippable
    eff_span : tuple[int,int] 

    # MethodEx indices already visited. Keep this list in case 
    #  for some reason we don't go through them in order
    visted_subtask_exec_indices : List[int]

    is_operator_frame : bool
    is_span_overflow : bool

    def __init__(self, task_exec, child_execs,
                child_exec_index, visited_child_exec_inds,
                subtask_exec_index, visted_subtask_exec_indices=[], eff_span=None):
        self.task_exec = task_exec
        self.child_execs = child_execs
        self.child_exec_index = child_exec_index
        self.visited_child_exec_inds = visited_child_exec_inds
        self.subtask_exec_index = subtask_exec_index
        self.visted_subtask_exec_indices = visted_subtask_exec_indices

        child = child_execs[child_exec_index] if child_exec_index < len(child_execs) else None
        self.is_operator_frame = isinstance(child, OperatorEx)

        if(eff_span is None and not self.is_operator_frame):
            self.eff_span = self._index_eff_span(child_exec_index)
        else:
            self.eff_span = eff_span

        self.is_span_overflow = (not self.is_operator_frame and 
                                 self.eff_span[1] >= len(child.subtask_execs))

    def _next_child_exec_index(self):
        # TODO : Might consider policies other than trying MethodExs sequentially  
        nxt = self.child_exec_index + 1 
        if(nxt >= len(self.child_execs)):
            return None
        return nxt

    def _index_eff_span(self, ind):
        ''' 
        For a particular subtask index, returns the effective span
        of other subtasks that are reachable within the same
        unordered group, or because subsequent subtasks are optional
        '''
        method_exec = self.current_method_exec
        method = method_exec.method
        spans = method.unord_spans
        start, end = ind, ind+(len(method.subtasks) > 0)
        # print()
        # print(ind, "SPANS", spans)
        for (s,e) in spans:
            if(s < end and e >= start):
                if(s < start): start = s
                if(e > end): end = e
            while(end < len(method.subtasks) and 
                  method.subtasks[end-1].optional == True):
                end += 1
        # print(start, end, len(method.subtasks))
        sbtsks = [method.subtasks[i] for i in range(start, end)]
        prnt = lambda x : x.name + ("*" if x.optional else "")
        # print(start, end, ", ".join([prnt(x) for x in sbtsks]))
        return start, end

    def _next_subtask_exec_index(self):

        # self._resolve_index_eff_span(self.subtask_exec_index)

        # TODO : Might consider policies other than trying TaskExs sequentially  
        nxt = self.subtask_exec_index + 1 
        subtask_execs = self.current_method_exec.subtask_execs
        if(nxt >= len(subtask_execs)):
            return None

        # self._index_eff_span(nxt)
        return nxt

    def next_child_frame(self):
        ind = self._next_child_exec_index()

        if(ind is None):
            return None

        return FrameContext(
            task_exec =                 self.task_exec,
            child_execs =     self.child_execs,
            child_exec_index =         ind, 
            visited_child_exec_inds = (
                self.visited_child_exec_inds + [self.child_exec_index],
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
            child_execs =               self.child_execs,
            child_exec_index =          self.child_exec_index, 
            visited_child_exec_inds = self.visited_child_exec_inds,
            subtask_exec_index =         ind,
            visted_subtask_exec_indices = (
                self.visted_subtask_exec_indices + [self.subtask_exec_index]
            ),
        )

    @classmethod
    def new_frame(self, task_exec, child_execs, 
                    child_index=0,
                    subtask_index=0):

        return FrameContext(
            task_exec =                  task_exec,
            child_execs =               child_execs,
            child_exec_index =          child_index, 
            visited_child_exec_inds = [],
            subtask_exec_index =         subtask_index,
            visted_subtask_exec_indices = [],
        )

    @property
    def current_task_exec(self):
        return self.task_exec

    @property
    def current_child_exec(self):
        child_execs = self.child_execs
        child_ind = self.child_exec_index
        if(child_execs and child_ind is not None 
            and len(child_execs) > 0):

            return child_execs[child_ind]
        else:
            return None

    @property
    def current_method_exec(self):
        child = self.current_child_exec
        assert(isinstance(child, MethodEx))
        return child

    @property
    def current_operator_exec(self):
        child = self.current_child_exec
        assert(isinstance(child, OperatorEx))
        return child

    @property
    def current_subtask_exec(self):
        if(self.is_operator_frame or not self.current_method_exec):
            return None
        subtask_execs = self.current_method_exec.subtask_execs
        ind = self.subtask_exec_index
        if(subtask_execs and ind is not None
           and len(subtask_execs) > 0):
            return subtask_execs[ind]
        else:
            return None

    def get_cand_next_subtask_execs(self):
        # s,e = self._index_eff_span(self.child_exec_index)
        s,e = self.eff_span

        subtask_execs = []
        method_exec = self.current_method_exec
        for i in range(s,e):
            subtask_execs.append(method_exec.subtask_execs[i])
        return subtask_execs

    @property
    def is_nomatch(self):
        return not self.child_execs

    def __str__(self):
        task_name = self.task_exec.task.name
        if(not self.is_operator_frame):
            s = ""
            start, end = self.eff_span
            subtask_execs = self.current_method_exec.subtask_execs
            for i in range(len(subtask_execs)):
                ex = subtask_execs[i]
                if(i == start):
                    s += "\033[92m"
                if(i == self.subtask_exec_index):
                    s += "|"
                s += ex.task.name
                if(ex.task.optional): s += "*"

                if(i == end-1):
                    s += "\033[0m"
                if(i != len(subtask_execs)-1):
                    s += ", "

            return f"{task_name} -> {s}"
        else:    
            return f"{self.current_operator_exec}"

    __repr__ = __str__


class Cursor:
    def __init__(self, trace=None):
        # Current task execution being processed
        self.trace = Trace() if trace is None else trace
        self.reset()

    def reset(self):
        self.current_frame = None
        self.stack = []

    def copy(self):
        new_cursor = Cursor()
        new_cursor.trace = self.trace
        new_cursor.stack = self.stack
        new_cursor.current_frame = self.current_frame
        return new_cursor

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

    @property
    def is_operator_frame(self):
        return self.current_frame.is_operator_frame

    def get_cand_next_subtask_execs(self):
        return self.current_frame.get_cand_next_subtask_execs()

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
            if(not curr_frame.is_operator_frame):
                if(curr_frame.current_subtask_exec):
                    curr_frame.current_subtask_exec.status = ExStatus.SUCCESS
                next_frame = self.current_frame.next_subtask_frame()
                if(next_frame is not None):
                    if(trace):
                        trace.add(TraceKind.ADVANCE_SUBTASK, curr_frame, next_frame)
                    break
            
                # If subtask sequence exhasted try popping frame from stack
                curr_frame.current_task_exec.status = ExStatus.SUCCESS
            if(curr_frame.current_child_exec):
                curr_frame.current_child_exec.status = ExStatus.SUCCESS
            next_frame = self._pop_frame()
            if(next_frame is None):
                break
            if(trace):
                trace.add(TraceKind.POP_FRAME, curr_frame, next_frame)
                

        self.current_frame = next_frame

    def push_task_exec(self, task_exec, child_execs, child_ind=0, trace=None):
        ''' Recurse into a new frame from a task execution and
            its list of method executions
        ''' 
        # Push current frame 
        if(self.current_frame is not None):
            self.stack.append(self.current_frame)

        # Make a new frame pointing at the selected MethodEx
        curr_frame = self.current_frame        
        next_frame = FrameContext.new_frame(task_exec, child_execs, child_ind)
        next_frame.current_child_exec.status = ExStatus.IN_PROGRESS
        next_frame.current_child_exec.status = ExStatus.IN_PROGRESS
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
        next_frame = None
        while True:
            curr_frame = self.current_frame
            if(curr_frame is not None):
                curr_frame.current_task_exec.status = ExStatus.FAILED
                if(curr_frame.current_method_exec):
                    curr_frame.current_method_exec.status = ExStatus.FAILED
                if(curr_frame.current_subtask_exec):
                    curr_frame.current_subtask_exec.status = ExStatus.FAILED
                next_frame = self._pop_frame()

            if(trace):
                trace.add(trace_kind, curr_frame, next_frame)

            if not next_frame:
                return False
            
            # Try next possible MethodEx
            curr_frame = self.current_frame
            next_frame = self.current_frame.next_child_frame()
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
            curr_frame.child_execs,
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
        if(task_frame.child_execs is None):
            task_frame.child_execs = []
        task_frame.child_execs.append(method_exec)

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

        print(f"Possible MethodExs: {len(frame.child_execs)}")
        print(f"Current Method Index: {frame.child_exec_index}")
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

    def __str__(self):
        return f"Cursor(depth={len(self.stack)}, {self.current_frame})"

    __repr__ = __str__


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
        self.cursors = [Cursor()]

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
        return self.root_task_queue.is_empty() and all(c.is_at_end() in self.cursors)

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




    def _expand_taskex(self, task_exec, cursor, push_nomatch_frames=False):
        ''' Handle the execution of a TaskEx instance
            :return: A TaskKind Enum signaling how the execution was carried out
        '''

        # Apply pattern matching to find child MethodExs or OperatorEx
        #   of the current TaskEx.
        # print("get_child_executions", task_exec)
        child_execs = task_exec.get_child_executions(
            self.domain_network, self.state
        )

        # Backtrack or push a nomatch frame if matching fails for any reason.
        if(child_execs is None or len(child_execs) == 0):
            if(push_nomatch_frames):
                print("NO MATCH FRAME", child_execs)
                cursor.push_nomatch_frame(task_exec, child_execs, trace=self.trace)
                return None, TraceKind.FIRST_SUBTASK
            else:
                
                backtrack_kind = (
                    TraceKind.BACKTRACK_NO_CHILDREN,
                    TraceKind.BACKTRACK_NO_MATCH
                )[child_execs is not None]

                cursor.backtrack(backtrack_kind, trace=self.trace)
                return None, backtrack_kind

        return child_execs, None

        
    def _pop_next_root_task(self, push_nomatch_frame):
        trace = self.cursors[0].trace
        cursor = Cursor(trace)
        root_task = self.root_task_queue.get_next_task()
        

        task_exec = root_task.as_task_exec(self.state)
        child_execs, trace_kind = self._expand_taskex(task_exec, cursor, push_nomatch_frame)
        cursor.push_task_exec(task_exec, child_execs, trace=trace)

        trace.add(TraceKind.NEW_ROOT_TASK, task_exec)
        if(trace_kind is not None):
            trace.add(trace_kind, task_exec)
        
        self.cursors = [cursor]


    def plan(self, multiheaded=False, stop_kinds=[], push_nomatch_frame=False) -> List:
        """
        Execute the planning process.
        :return: The complete plan.
        """
        self._plan_start()

        

        while True:
            # If all cursors exhausted, create a new cursor by popping off a new root task
            if(all(cursor.is_at_end() for cursor in self.cursors) and
               not self.root_task_queue.is_empty()):
                self._pop_next_root_task(push_nomatch_frame)

            # If root tasks all exhausted terminate
            else:
                self.trace = cursors[0].trace
                self.trace.add(TraceKind.ROOT_TASKS_EXHAUSTED)
                return trace


            # Plan all cursors down to their next stop point 
            #  or to their next operator execution, whatever comes first
            halted_cursors = []
            operator_cursors = []
            cursors = [*self.cursors]
            while(len(cursors) > 0):
                print("CURSORS:", cursors)
                cursor = cursors.pop()
                new_cursors = [] if multiheaded else [cursor]
                    
                # If the current cursor is pointing at an operator exec, save it
                if(cursor.is_operator_frame):
                    operator_cursors.append(cursor)
                    continue

                
                if(multiheaded):
                    # Go through the next subtasks pointed to by the current frame
                    for task_exec in cursor.get_cand_next_subtask_execs():
                        # print(">>", task_exec)
                        child_execs, trace_kind = self._expand_taskex(task_exec, cursor, push_nomatch_frame)

                        if(child_execs is None): continue
                            
                        for child_exec in child_execs:
                            # if(isinstance(child_exec, MethodEx)):
                            #     print("--", child_exec, child_exec.subtask_execs)
                            # else:
                            #     print("--", child_exec)
                            new_cursor = cursor.copy()
                            new_cursor.push_task_exec(task_exec, [child_exec], trace=cursor.trace)
                            trace_kind = TraceKind.EXPAND_TO_METHOD if isinstance(child_exec, MethodEx) else TraceKind.EXPAND_TO_OPERATOR
                            # print(trace_kind)
                            if(trace_kind in stop_kinds or trace_kind is None):
                                halted_cursors.append(new_cursor)
                            else:
                                new_cursors.append(new_cursor)
                        task_exec.status = ExStatus.IN_PROGRESS

                else:
                    task_exec = cursor.current_subtask_exec
                    child_execs, trace_kind = self._expand_taskex(task_exec, cursor, push_nomatch_frame)

                    if(child_execs is None): continue
                        
                    trace_kind = cursor.push_task_exec(task_exec, [child_exec], trace=cursor.trace)
                    task_exec.status = ExStatus.IN_PROGRESS

                    if(trace_kind in stop_kinds or trace_kind is None):
                        should_break = True

                        # trace_kind = self._step_cursor(task_exec, cursor, stop_kinds, push_nomatch_frame)
                cursors += new_cursors
                

            print("OPERATOR CURSORS:", [c.current_frame.current_operator_exec for c in operator_cursors])
            print("HALT CURSORS:", len(halted_cursors))

            if(len(halted_cursors) > 0):
                self.cursors = halted_cursors
                break; 

            if(self.env is not None):
                cursor = operator_cursors[0]
                self._handle_operator_exec(cursor)
                self.cursors = cursor
                break;


        # if(not multiheaded):
        #     while True:
        #         # TODO: Is there a case where cursors[0] resolves to something different or can we 
        #         #   grab this outside of the loop?
        #         cursor = self.cursors[0]
        #         trace_kind = self._step_cursor_linear(cursor, stop_kinds, push_nomatch_frame)
        #         if(trace_kind in stop_kinds or trace_kind is None):
        #             return self.trace

        # else:
        #     # TODO: This needs to work even if the cursor list changes while looping
        #     #   we might for instance advance one cursor in a way that invalidates others
        #     #   or add new cursors as we bifurcate the path.
        #     for cursor in self.cursors:
        #         while True:
        #             trace_kind = self._step_cursor_multiheaded(cursor, stop_kinds, push_nomatch_frame)
        #             if(trace_kind in stop_kinds or trace_kind is None):
        #                 break


        # for cursor in self.cursors:

        # Main Plan Loop:
        #  Each loop handles one TaskEx instance and makes one move of a Cursor.
        #   (the Cursor holds the current plan state and stack).
        #  Every Cursor movement is gaurenteed to either:
        #   1. Enter a valid frame w/ a parent TaskEx, MethodEx, and current sub-TaskEx.
        #   2. Exhaust the cursor (the current TaskEx will come from the root_task_queue)
            # while True:

                # self._step_cursor_linear

                # task_exec = self._next_task_exec()

                # if(task_exec is None):
                #     break

                # # Handle the current task execution
                # trace_kind = self._handle_task_exec(task_exec, cursor, push_nomatch_frame)
                # if(trace_kind in stop_kinds):
                #     return self.trace

        return self.trace

    def _handle_operator_exec(self, cursor):
        operator_exec = cursor.current_frame.current_operator_exec
        success = self._apply_operator_execution(operator_exec)
        if(success):
            cursor.trace.add(TraceKind.APPLY_OPERATOR, operator_exec)
            if(cursor.current_frame):
                cursor.advance_subtask(trace=cursor.trace)
            return TraceKind.ADVANCE_SUBTASK
        else:
            cursor.backtrack(
                TraceKind.BACKTRACK_OPERATOR_FAIL,
                trace=cursor.trace
            )
            return TraceKind.BACKTRACK_OPERATOR_FAIL


    def _next_task_exec(self, cursor):
        # If cursors are exhausted, pop off new root task, and make it into a TaskEx.
        if(cursor.is_at_end()):
            if(self.root_task_queue.is_empty()):
                self.trace.add(TraceKind.ROOT_TASKS_EXHAUSTED)
                return None

            root_task = self.root_task_queue.get_next_task()
            task_exec = root_task.as_task_exec(self.state)
            self.trace.add(TraceKind.NEW_ROOT_TASK, task_exec)

        # Otherwise find all the next TaskExs that the cursor is pointing to
        else:
            task_exec = cursor.current_subtask_exec

        return task_exec

    # def _step_cursor(self, task_exec, cursor, stop_kinds, push_nomatch_frame):
    #     if(task_exec is None):
    #         return None

    #     # Expand the current task execution
    #     child_execs, trace_kind = self._expand_taskex(task_exec, cursor, push_nomatch_frame)

    #     if(child_execs is None):
    #         return trace_kind

    #     # Primitive Task Case: execute Operator
    #     if(isinstance(child_execs[0], OperatorEx)):
    #         trace_kind = self._handle_operator_exec(cursor, child_execs[0])

    #     # Higher-Order Task Case: Push a new frame with options for 
    #     #    executing the task (will be handled in next loop).
    #     else:
    #         cursor.push_task_exec(task_exec, child_execs, trace=self.trace)
    #         task_exec.status = ExStatus.IN_PROGRESS
    #         trace_kind = TraceKind.FIRST_SUBTASK

    #     return trace_kind


    # def _all_next_task_execs(self, cursor):
        


    # def _step_cursor_multiheaded(self, cursor, stop_kinds, push_nomatch_frame):
    #     if(cursor.is_at_end()):
    #         return None
        
    #     task_execs = cursor.get_cand_next_subtask_execs()

    #     # print(cursor)
    #     if(task_execs is None or len(task_execs) == 0):
    #         return None

    #     pairs = []
    #     should_branch = False
    #     trace_kind = None
    #     pref = [None, TraceKind.BACKTRACK_NO_CHILDREN, TraceKind.BACKTRACK_NO_MATCH]

    #     for task_exec in task_execs:
    #         # Expand the current task execution
    #         child_execs, _trace_kind = self._expand_taskex(task_exec, cursor, push_nomatch_frame)


    #         if(pref.index(_trace_kind) > pref.index(trace_kind)):
    #             trace_kind = _trace_kind

    #         if(child_execs is None):
    #             continue

    #         # If the current frame has any staged operator executions then if 
    #         #   there are also any staged method executions those should be handled 
    #         #   by copies of the current cursor.
    #         if(any(isinstance(child_exec, OperatorEx) for child_exec in child_execs)):
    #             should_branch = True

    #         pairs.push(task_exec, child_execs)

    #     if(len(pairs) == 0):
    #         return trace_kind

    #     for task_exec, child_execs in pairs:
    #         for child_exec in child_execs:
    #             # Primitive Task Case: execute Operator
    #             if(isinstance(child_exec, OperatorEx)):
    #                 trace_kind = self._handle_operator_exec(cursor, child_exec)

    #             # Higher-Order Task Case: Push a new frame with options for 
    #             #    executing the task (will be handled in next loop).
    #             else:
    #                 c = cursor
    #                 if(should_branch):
    #                     c = cursor.copy()
    #                     self.cursors.append(c)
    #                 c.push_task_exec(task_exec, [child_exec], trace=self.trace)
    #                 task_exec.status = ExStatus.IN_PROGRESS
    #                 trace_kind = TraceKind.FIRST_SUBTASK
    #                 should_branch = True

    #     return trace_kind




            

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

        cursor = self.cursors[0]

        print(f"Current task: {cursor.current_task.name if cursor.current_task else None}")
        print(f"Current method: {cursor.current_method.name if cursor.current_method else None}")
        print(f"Current subtask index: {cursor.current_subtask_index}")
        print(f"Available methods: {len(cursor.available_method_execs)}")
        print(f"Current method index: {cursor.current_method_index}")
        print(f"Stack depth: {len(cursor.stack)}")

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
        method_execs = frame.child_execs
        return frame.task_exec, method_execs


    def plan_to_next_operator(self, push_nomatch_frame=True):
        return self.plan(stop_kinds=[TraceKind.ADVANCE_SUBTASK], push_nomatch_frame=push_nomatch_frame)

    def get_prev_operator_exec(self):
        op_exec = self.trace.get_prev_operator()
        return op_exec 

    def stage_method_exec(self, method_exec):
        task_exec, method_execs = self.get_next_method_execs()
        inds = [i for i, x in enumerate(method_execs) if x == method_exec]

        if(len(inds) == 0):
            raise ValueError(f"MethodEx stage {method_exec} fail." +
                "Not a possible MethodEx for this state")

        self.cursor.user_select_method_exec(inds[0], self.trace)
    
    def add_method_exec(self, method_exec, task_exec=None):
        self.cursor.add_method_exec(method_exec, task_exec, self.trace)


    ####################
    # HELPER FUNCTIONS #
    ####################

    def _apply_operator_execution(self, operator_exec: OperatorEx) -> bool:
        """Execute an operator and update state."""

        # Execute in environment if available
        # print(operator_exec)

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
                print(">> Exception")
                print(e)
                raise e
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
            

    # def _get_method_executions(self) -> List[MethodEx]:
    #     """Get methods applicable to the current task."""
    #     if self.enable_logging:
    #         self.logger.log_function()
    #         self.logger.info(f"Locating applicable methods for task {self.cursor.current_task.name}")

    #     if not self.state:
    #         raise ValueError("State must be set before getting applicable methods")

    #     # Check if there are methods for this task type
    #     if self.cursor.current_task.id not in self.domain_network:
    #         if self.enable_logging:
    #             self.logger.warning(f"No methods found for task {self.cursor.current_task.id}")
    #         return []

    #     applicable_methods = []
    #     methods = self.domain_network[self.cursor.current_task.id]

    #     # Sort by cost if requested
    #     if self.order_by_cost:
    #         methods = sorted(methods, key=lambda m: m.cost)

    #     # Check each method
    #     visited = []
    #     for method in methods:
    #         result = method.applicable(self.cursor.current_task, self.state, str(self.trace.get_current_plan()), visited)
    #         if result:
    #             applicable_methods.append(result)
    #             if self.enable_logging:
    #                 self.logger.info(f"Method {method.name} is applicable to task {self.cursor.current_task.name}")

    #     if self.enable_logging:
    #         self.logger.info(
    #             f"Found {len(applicable_methods)} applicable methods for task {self.cursor.current_task.name}")

    #     return applicable_methods

    # def _process_current_task(self) -> bool:
    #     """Process the current task by finding and applying methods."""
    #     if self.enable_logging:
    #         self.logger.log_function()
    #         self.logger.info(f"Processing task {self.cursor.current_task.name}")

    #     # Add task entry to trace
    #     self.trace.add_task(self.cursor.current_task)

    #     # If we already have a method, continue executing it
    #     if self.cursor.current_method:
    #         return self._continue_method_execution()

    #     # Get applicable methods for this task
    #     method_execs = self._get_method_executions()
    #     if not method_execs:
    #         if self.enable_logging:
    #             self.logger.warning(f"No applicable methods for task {self.cursor.current_task.name}")
    #         self.cursor.current_task.status = 'failed'
    #         return False

    #     # Store available methods for backtracking
    #     self.cursor.available_method_execs = method_execs
    #     self.cursor.current_method_index = 0
    #     self.cursor.current_method = methods[0]
    #     self.cursor.current_subtask_index = 0

    #     # Add method entry to trace
    #     self.trace.add_method(
    #         self.cursor.current_task,
    #         self.cursor.current_method,
    #         "Method selected as first applicable method"
    #     )

    #     if self.enable_logging:
    #         self.logger.info(
    #             f"Selected method {self.cursor.current_method.name} for task {self.cursor.current_task.name}")

    #     # Execute the selected method
    #     return self._continue_method_execution()

    # def _set_cursor_to_new_task(self):
    #     if self.enable_logging:
    #         self.logger.log_function()
    #     next_task = self.root_task_execs.pop(0)
    #     if self.enable_logging:
    #         self.logger.info(f"Moving on to new task: ({next_task.name})")
    #     # Handle repeat tasks
    #     if next_task.repeat > 0:
    #         next_task.repeat -= 1
    #         self.root_task_execs.insert(0, next_task)
    #     self.cursor.set_task(next_task)
    #     # Get and store applicable methods
    #     method_execs = self._get_method_executions()
    #     if not method_execs:
    #         if self.enable_logging:
    #             self.logger.info(f"No applicable methods found for task: ({self.cursor.current_task.name})")
    #         return False

    #     self.cursor.available_method_execs = methods
    #     self.cursor.current_method_index = 0
    #     return True





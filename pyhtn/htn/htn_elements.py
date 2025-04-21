from pyhtn.common.imports.typing import *
from abc import ABC, abstractmethod
from pyhtn.domain.variable import Var
from pyhtn.common.utils import rand_uid


from pyhtn.conditions.fact import Fact
from pyhtn.conditions.conditions import NOT


from pyhtn.conditions.pattern_matching import (
    dict_to_tuple,
    fact_to_tuple,
    msubst,
    subst,
    tuples_to_dicts,
    unify,
)

from py_plan.pattern_matching import build_index, pattern_match



# ------------------------------------------------------
# : HTN_Element

class HTN_Element(ABC):
    def __init__(self,
                 name: str,
                 args: Sequence[Union[Var, Any]] = (),
                 cost=1.0) -> None:

        
        self.name = name
        self.args = tuple(args)
        self.cost = cost

    @abstractmethod
    def __str__(self):
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError('Subclasses must implement this method.')

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

class MatchableMixin(ABC):
    def _get_match_substitutions(self, task_exec, state):
        ptstate = dict_to_tuple(state)
        index = build_index(ptstate)
        substitutions = unify(task_exec.match, self.args)
        # print(task_exec.match, self.args)

        # Find the substitutions for each match
        if(not self.preconditions):
            match_substs = [substitutions] 
        else:
            # Danny Question: When would there ever be multiple self.preconditions?
            ptcondition = fact_to_tuple(self.preconditions, variables=True)[0]
            match_substs = [x for x in pattern_match(ptcondition, index, substitutions)]

        return match_substs

    # @abstractmethod
    # def get_match_executions(self, task_exec, state):
    #     raise NotImplementedError('Subclasses must implement this method.')



class Operator(HTN_Element, MatchableMixin):
    ''' TODO: Describe 
    
    -- Examples --

    Args can be defined positionally or as a keyword argument:
        Operator("make_hat", V('hat'), effects=[Fact(hat=V('hat'))] )
        Operator("make_hat", args=[V('hat')], effects=[Fact(hat=V('hat'))] )
    '''
    def __init__(self,
                 name : str = None, 
                 *_args : Var, # Positional 
                 args: Sequence[Var] = (), # 
                 effects: Sequence[Fact] = (),
                 preconditions=None,
                 cost=1.0) -> None:

        if(len(args) == 0):
            args = _args

        super().__init__(name, args, cost)
        self.id = f"O_{rand_uid()}"

        if(len(args) == 0):
            args = _args

        self.effects = effects
        self.preconditions = preconditions

        self.add_effects = set()
        self.del_effects = set()

        if isinstance(self.effects, Fact):
            self.add_effects.add(self.effects)
        elif isinstance(self.effects, NOT):
            self.del_effects.add(self.effects[0])
        else:
            for e in self.effects:
                if isinstance(e, NOT):
                    self.del_effects.add(e[0])
                else:
                    self.add_effects.add(e)

    def __str__(self):
        if(self.name is not None):
            prefix = f"Operator({self.name!r}"
        else:
            prefix = f"Operator("

        # print(prefix, self.args)

        if(len(self.args) > 0):
            return f"{prefix}, {', '.join([repr(x) for x in self.args])})"
        else:
            return f"{prefix}{self.name!r})"

    __repr__ = __str__

    def get_match_executions(self, task_exec, state):
        from pyhtn.htn.element_executions import OperatorEx

        match_substs = self._get_match_substitutions(task_exec, state)

        # Make Method exections and Task executions from each match.
        op_execs = []
        for m_subst in match_substs:
            op_execs.append(
                OperatorEx(
                    self, state,
                    subst(m_subst, self.args),
                    parent_task_exec=task_exec,
                )
            )
            
        return op_execs




# ------------------------------------------------------
# : Task

class Task(HTN_Element):
    def __init__(self,
                 name: str,
                 *_args : Union[Var, Any],
                 args: Sequence[Union[Var, Any]] = (),
                 cost=1.0,
                 priority='first',
                 repeat=1):

        if(len(args) == 0):
            args = _args

        super().__init__(name, args, cost)
        self.id = f"T_{rand_uid()}"

        self.domain_key = f'{name}/{len(args)}'
        self.priority = priority
        self.repeat = repeat

    def as_task_exec(self, state):
        '''Create a task execution from a task by taking its
            arguments as its match and fixing it to a state.
        '''
        from pyhtn.htn.element_executions import TaskEx
        return TaskEx(self, state, self.args)

    def __str__(self):
        if(len(self.args) > 0):
            return f"Task({self.name!r}, {', '.join([repr(x) for x in self.args])})"
        else:
            return f"Task({self.name!r})"

    __repr__ = __str__

    def get_child_executions(self, domain, state):
        task_exec = self.as_task_exec(state)
        return task_exec.get_child_executions(domain, state)

# ------------------------------------------------------
# : Method

class Method(HTN_Element, MatchableMixin):
    def __init__(self,
                 name: str,
                 *_args : Var, # Positional 
                 args: Sequence[Union[Var, Any]] = (),
                 subtasks: Sequence[Union[Task, tuple]] = [],
                 preconditions=None,
                 cost=1.0) -> None:

        if(len(args) == 0):
            args = _args

        super().__init__(name, args, cost)
        self.id = f"M_{rand_uid()}"

        self.subtasks = subtasks
        self.preconditions = preconditions

    @property
    def task():
        if(not hasattr(self, '_task')):
            raise RuntimeError((
                f"Method was defined with name={self.name} " 
                "but has not resolved its singleton Task object "
                "from the domain."
            ))
        return self._task


    def __str__(self):
        if(len(self.args) > 0):
            s = f"Method({self.name!r}, {', '.join([repr(x) for x in self.args])}"
        else:
            s = f"Method({self.name!r}"

        s += f", subtasks={self.subtasks})"
        return s

    __repr__ = __str__


    def get_match_executions(self, task_exec, state):

        ''' Get all Method exectutions that match the Method's preconditions
            given a parent Task execution.
        ''' 

        from pyhtn.htn.element_executions import MethodEx, TaskEx

        match_substs = self._get_match_substitutions(task_exec, state)

        # Make Method exections and Task executions from each match.
        meth_execs = []
        for m_subst in match_substs:
            meth_exec = MethodEx(
                self, state,
                subst(m_subst, self.args),
                parent_task_exec=task_exec,
            )
            subtask_execs = []
            for subtask in self.subtasks:
                subtask_execs.append(
                    TaskEx(
                        subtask, state,
                        subst(m_subst, subtask.args),
                        parent_method_exec=meth_exec,
                    )
                )
            meth_exec.child_execs = subtask_execs
            meth_execs.append(meth_exec)

        return meth_execs


            
            


        
        # matches = [ for theta in ]

        # for ptcondition in ptconditions:
        #     matches = [(self.name, theta) for theta in pattern_match(ptcondition, index, substitutions)]
        # for theta in 
            # grounded_subtasks = self._create_grounded_subtasks(msubst(theta, self.subtasks))
            # matched_facts = tuples_to_dicts(subst(theta, tuple(ptcondition)), use_facts=True, use_and_operator=True)
            # return GroundedMethod(name=self.name,
            #                       subtasks=grounded_subtasks,
            #                       matched_facts=matched_facts,
            #                       args=self.args,
            #                       preconditions=self.preconditions,
            #                       cost=self.cost,
            #                       parent_id=self.id)


        # print(subtask_matches)
        # for subtask in self.subtasks:
        #     st_ex = TaskExecution(
        #         subtask,
        #         subst(substitutions, tasks.args)
        #     )
        #     subtask_execs.append(st_ex)


        # subst(theta, tasks.args)

        # for 

        # print(msubst(the, self.subtasks))


        # if not self.preconditions:
        #     subtask_executions = self._create_grounded_subtasks(msubst(substitutions, self.subtasks))
        #     print(grounded_subtasks)
        # else:
        #     ptconditions = fact_to_tuple(self.preconditions, variables=True)

        #     return GroundedMethod(name=self.name,
        #                           subtasks=grounded_subtasks,
        #                           matched_facts=[],
        #                           args=self.args,
        #                           preconditions=self.preconditions,
        #                           cost=self.cost,
        #                           parent_id=self.id)


        # ptconditions = fact_to_tuple(self.preconditions, variables=True)
        # for ptcondition in ptconditions:
        #     matches = [(self.name, theta) for theta in pattern_match(ptcondition, index, substitutions)]
        #     if matches:
        #         method_name, theta = choice(matches)
        #         grounded_subtasks = self._create_grounded_subtasks(msubst(theta, self.subtasks))
        #         matched_facts = tuples_to_dicts(subst(theta, tuple(ptcondition)), use_facts=True, use_and_operator=True)
        #         return GroundedMethod(name=self.name,
        #                               subtasks=grounded_subtasks,
        #                               matched_facts=matched_facts,
        #                               args=self.args,
        #                               preconditions=self.preconditions,
        #                               cost=self.cost,
        #                               parent_id=self.id)












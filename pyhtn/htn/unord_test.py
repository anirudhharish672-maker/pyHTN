from pyhtn.htn import Method, Operator, Task, MethodEx, OperatorEx, TaskEx, Unord
from pyhtn.htn.element_executions import tree_dict_to_str
from pyhtn.htn.planner2 import HtnPlanner2
from pyhtn.conditions.fact import Fact
from pyhtn.conditions.conditions import NOT
from pyhtn.domain.variable import V
from pprint import pprint
from pyhtn.planner.planner import HtnPlanner
from unittest.mock import MagicMock, patch
from pprint import pprint


operators = {
    "a" : [
        Operator('a', effects=[Fact(id='a', did_it=True)]),
    ],

    "b" : [
        Operator('b', effects=[Fact(id='b', did_it=True)]),
    ],

    "c" : [
        Operator('c', effects=[Fact(id='c', did_it=True)]),
    ],

    "d" : [
        Operator('d', effects=[Fact(id='d', did_it=True)]),
    ],

    "e" : [
        Operator('e', effects=[Fact(id='e', did_it=True)]),
    ],

    "f" : [
        Operator('f', effects=[Fact(id='f', did_it=True)]),
    ],

    "g" : [
        Operator('g', effects=[Fact(id='g', did_it=True)]),
    ],

    "x" : [
        Operator('x', effects=[Fact(id='x', did_it=True)]),
    ],

    "y" : [
        Operator('y', effects=[Fact(id='y', did_it=True)]),
    ],

    "z" : [
        Operator('z', effects=[Fact(id='z', did_it=True)]),
    ]
}






def setup(domain):
    env = MagicMock()
    env.get_state.return_value = [{
        'id': '1', 
        'skip_A': True, "skip_B" : False, 
        'opt_C': True, "opt_D" : False, 
    }]
    planner = HtnPlanner2(
        tasks = [{'name': 'S', 'args': []}],
        domain = domain,
        env = env,
        enable_logging=True
    )
    return planner

def plan_next(planner):
    trace = planner.plan_to_next_operators(multiheaded=True)
    planner.print_current_frames()
    print()

def check_next(planner, op_names, nxt_op):
    operator_execs = planner.get_next_operator_execs()
    res_op_names = [op_ex.operator.name for op_ex in operator_execs]
    assert res_op_names == op_names, f"{res_op_names} != {op_names}"
    planner.apply(operator_execs[op_names.index(nxt_op)])
    return operator_execs

def check_exhausted(planner):
    assert planner.is_exhausted()
    planner.print_current_frames()


domain0 = {
    **operators,

    "S" : [
        Method('S',
            subtasks=[
                Task("a", optional=True),
                Unord(
                    Task('b', optional=True),
                    Task('c', optional=True),
                    Task('d', optional=True),
                ),
                Unord(
                    Task('e', optional=True),
                    Task('f'),
                ),
                Task('g'),
            ],
        )
    ],
}

def test_basic_unord_skip():
    print("--------------")
    planner = setup(domain0)
    plan_next(planner)
    check_next(planner, ['a', 'b', 'c', 'd', 'e', 'f'], 'd')
    plan_next(planner)
    check_next(planner, ['b', 'c', 'e', 'f'], 'f')
    plan_next(planner)

domain1 = {
    #  ----  OPERATORS -------    
    **operators,

    # ----- METHODS ----------
    "S" : [
        Method('S',
            subtasks=[
                Task("U"),
                Task('f', optional=True),
                Task('g'),
            ],
        )
    ],

    "U" : [
        Method('U',
            subtasks=[
                Task('a', optional=True),
                Task('Q'),
                Task('B'),
                Unord(
                    Task('c'),
                    Task('d'),
                    Task('e', optional=True),
                ),
                
            ],
        )
    ],


    "Q" : [
        # Tests that committing to a method narrows executable operators
        #   to the scope of that method (and its descendants).
        Method('Q', subtasks=[Task('b'), Task('T')]),

        # Test that skippabilitiy propogates up multiple levels
        Method('Q', subtasks=[Task('R')]),
    ],
    "R" : [
        Method('R',subtasks=[]),
    ],
    "T" : [
        Method('S',subtasks=[Task("z")]),
    ],

    # Tests that skipped optionals propogate effective span into the parent
    "B" : [
        Method('B', subtasks=[Task('x'), Task('y', optional=True)]),
        Method('B',subtasks=[]),
    ],   
}

def test_heir_unord_skip():
    print("--------------")
    planner = setup(domain1)
    plan_next(planner)
    check_next(planner, ['a', 'b', 'x', 'c', 'd', 'e'], 'b')
    plan_next(planner)
    check_next(planner, ['z'], 'z')
    plan_next(planner)
    check_next(planner, ['x', 'c', 'd', 'e'], 'x')
    plan_next(planner)
    check_next(planner, ['y', 'c', 'd', 'e'], 'y')
    plan_next(planner)
    check_next(planner, ['c', 'd', 'e'], 'd')
    plan_next(planner)
    check_next(planner, ['c', 'e'], 'c')
    plan_next(planner)
    check_next(planner, ['e', 'f', 'g'], 'f')
    plan_next(planner)
    check_next(planner, ['g'], 'g')
    check_exhausted(planner)



domain2 = {
    #  ----  OPERATORS -------    
    **operators,

    # ----- METHODS ----------
    

    "S" : [
        Method('S',
            subtasks=[
                Task("A"),
                Task('B'),
                Task('C'),
                Task('D'),
                Task('e'),
            ],
        )
    ],
    "A" : [
        Method('A', preconditions=[Fact(skip_A=False)], 
            subtasks=[Task('a')]),
        Method('A', preconditions=[Fact(skip_A=True)],
            subtasks=[]),
    ],
    "B" : [
        Method('B', preconditions=[Fact(skip_B=False)], 
            subtasks=Unord(Task('z',optional=True), Task('b'))),
        Method('B', preconditions=[Fact(skip_B=True)],
            subtasks=[]),
    ],
    "C" : [
        Method('C', preconditions=[], 
            subtasks=[Task('c')]),
        Method('C', preconditions=[Fact(opt_C=True)],
            subtasks=[]),
    ],
    "D" : [
        Method('D', preconditions=[], 
            subtasks=[Task('d')]),
        Method('D', preconditions=[Fact(opt_D=True)],
            subtasks=[]),
    ],
}

def test_conditional_empty_methods():
    print("--------------")
    planner = setup(domain2)
    plan_next(planner)
    check_next(planner, ['z','b'], 'b')
    plan_next(planner)
    check_next(planner, ['z', 'c','d'], 'd')
    plan_next(planner)
    check_next(planner, ['e'], 'e')
    check_exhausted(planner)


    # trace = planner.plan_to_next_operators(multiheaded=True)
    # print("------------")
    # planner.print_current_frames()
    # # trace.print_trace()
    # operator_execs = planner.get_next_operator_execs()
    # print([op_ex.operator.name for op_ex in operator_execs])

    # planner.apply(operator_execs[1])

    # trace = planner.plan_to_next_operators(multiheaded=True)
    # print("------------")
    # planner.print_current_frames()
    # # trace.print_trace()
    # operator_execs = planner.get_next_operator_execs()
    # print([op_ex.operator.name for op_ex in operator_execs])

    # planner.apply(operator_execs[0])

    # trace = planner.plan_to_next_operators(multiheaded=True)
    # print("------------")
    # planner.print_current_frames()
    # # trace.print_trace()
    # operator_execs = planner.get_next_operator_execs()
    # print([op_ex.operator.name for op_ex in operator_execs])

    # planner.apply(operator_execs[1])

    # trace = planner.plan_to_next_operators(multiheaded=True)
    # print("------------")
    # planner.print_current_frames()
    # # trace.print_trace()
    # operator_execs = planner.get_next_operator_execs()
    # print([op_ex.operator.name for op_ex in operator_execs])

    # planner.apply(operator_execs[0])

    # trace = planner.plan_to_next_operators(multiheaded=True)
    # print("------------")
    # planner.print_current_frames()
    # # trace.print_trace()
    # operator_execs = planner.get_next_operator_execs()
    # print([op_ex.operator.name for op_ex in operator_execs])

    # planner.apply(operator_execs[1])

    # planner.print_current_frames()

    # trace = planner.plan_to_next_operators(multiheaded=True)
    # print("------------")
    # planner.print_current_frames()
    # # trace.print_trace()
    # operator_execs = planner.get_next_operator_execs()
    # print([op_ex.operator.name for op_ex in operator_execs])



    
if __name__ == "__main__":
    test_basic_unord_skip()
    test_heir_unord_skip()
    test_conditional_empty_methods()



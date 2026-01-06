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



def setup(domain, state=[{"id" : "1"}], args=[]):
    env = MagicMock()
    env.get_state.return_value = state
    planner = HtnPlanner2(
        tasks = [{'name': 'S', 'args': args}],
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
    check_next(planner, ['e', 'g'], 'g')
    check_exhausted(planner)

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
    state = [{
        'id': '1', 
        'skip_A': True, "skip_B" : False, 
        'opt_C': True, "opt_D" : False, 
    }]
    planner = setup(domain2, state)
    plan_next(planner)
    check_next(planner, ['z','b'], 'b')
    plan_next(planner)
    check_next(planner, ['z', 'c','d'], 'd')
    plan_next(planner)
    check_next(planner, ['e'], 'e')
    check_exhausted(planner)


domain3 = {
    #  ----  OPERATORS -------    
    **operators,

    # ----- METHODS ----------
    "S" : [
        Method('S',
            subtasks=[
                Task('a', optional_if=[Fact(opt_a=True)]),
                Task('b', optional_if=[Fact(opt_b=True)]),
                Task('D'),
                Task('e'),
                Task('F', optional_if=[Fact(opt_F=True)]),
                Task('g'),
            ],
        )
    ],
    "D" : [
        Method('D', subtasks=[
                    Task('x', optional_if=[Fact(opt_x=True)]),
                    Task('y', optional_if=[Fact(opt_y=True)])
        ]),
        Method('D', subtasks=[Task('z')]),
    ],
    "F" : [
        Method('F', subtasks=[Task('f')])
    ]
}

def test_optional_if():
    state = [{
        'id' : "1",
        'opt_a': True, "opt_b" : False, 
        'opt_F': True, 
        "opt_x" : False, "opt_y" : True, 
    }]
    planner = setup(domain3, state)
    plan_next(planner)
    check_next(planner, ['a','b'], 'b')
    plan_next(planner)
    check_next(planner, ['z', 'x'], 'x') # TODO: out of order
    plan_next(planner)
    check_next(planner, ['y', 'e'], 'e')
    plan_next(planner)
    check_next(planner, ['f', 'g'], 'f')
    plan_next(planner)
    check_next(planner, ['g'], 'g')
    check_exhausted(planner)


domain4 = {
    #  ----  OPERATORS -------    
    **operators,

    # ----- METHODS ----------
    "S" : [
        Method('S',
            subtasks=[
                Task('a', skip_if=[Fact(skip_a=True)]),
                Task('b', skip_if=[Fact(skip_b=True)]),
                Task('D'),
                Task('e'),
                Task('F', skip_if=[Fact(skip_F=True)]),
                Task('g'),
            ],
        )
    ],
    "D" : [
        Method('D', subtasks=[
                    Task('x', skip_if=[Fact(skip_x=True)]),
                    Task('y', skip_if=[Fact(skip_y=True)])
        ]),
        Method('D', subtasks=[Task('z')]),
    ],
    "F" : [
        Method('F', subtasks=[Task('f')])
    ]
}

def test_skip_if():
    state = [{
        'id' : "1",
        'skip_a': True, "skip_b" : False, 
        'skip_F': True, 
        "skip_x" : False, "skip_y" : True, 
    }]
    planner = setup(domain4, state)
    plan_next(planner)
    check_next(planner, ['b'], 'b')
    plan_next(planner)
    check_next(planner, ['z', 'x'], 'x') # TODO: out of order
    plan_next(planner)
    check_next(planner, ['e'], 'e')
    plan_next(planner)
    check_next(planner, ['g'], 'g')
    check_exhausted(planner)



rewrite_using_rule = lambda x : x
multiply_exponents = lambda x : x
solve_using_power_rule = lambda x : x
import re

domain_al = {
    'done':
        Operator(
            name=('done'),
            args=[V('kc')],
            # preconditions=[
            #     Fact(start=True),
            # ],
            effects=[
                Fact(
                    field='done',
                    value=((re.compile('x'),),),
                    kc=V('kc'),
                    answer=True
                ),
            ],
        ),

    'rewrite_using_rule':
        Operator(
            name=('rewrite_using_rule'),
            args=[V('equation'), V('kc')],
            preconditions=[],
            effects=[
                Fact(
                    field='rewrite_using_rule',
                    value=(rewrite_using_rule, V('eq')),
                    kc=V('kc'),
                    answer=True
                ),
            ],
        ),

    'multiply_exponents':
        Operator(
            name=('multiply_exponents'),
            args=[V('equation'), V('kc')],
            preconditions=[],
            effects=[
                Fact(
                    field='multiply_exponents',
                    value=(multiply_exponents, V('eq')),
                    kc=V('kc'),
                    answer=True
                ),
            ],
        ),

    'solve_using_power_rule':
        Operator(
            name=('solve_using_power_rule'),
            args=[V('equation'), V('kc')],
            preconditions=[],
            effects=[
                Fact(
                    field='solve_using_power_rule',
                    value=(solve_using_power_rule, V('eq')),
                    kc=V('kc'),
                    answer=True
                ),
            ],
        ),
    'rewrite_and_multiply': [
        Method(
            name=('rewrite_and_multiply'),
            args=[V('equation')],
            preconditions=[],
            subtasks=[
                Unord(
                    Task(
                        name=('rewrite_using_rule'),
                        args=[V('equation'), ('rewrite_using_rule',)],
                    ),
                    Task(
                        name=('multiply_exponents'),
                        args=[V('equation'), ('multiply_exponents',)],
                    ),
                )
            ],
        ),
    ],
    
    'S': [
        Method(
            name=('S'),
            args=[V('equation')],
            preconditions=[],
            subtasks=[
                Unord(
                    Task(
                        name='rewrite_and_multiply',
                        args=[V('equation')],
                        skip_if=[Fact(rewrite_and_multiply=False)],
                    ),
                    Task(
                        name='solve_using_power_rule',
                        args=[V('equation'), ('solve_using_power_rule',)],
                    ),
                ),
                Task(
                    name=('done'),
                    args=[('done',)],
                ),
            ],
        ),
    ],
}

def test_al_domain():
    state = [{
        'id' : "1",
        'start' : True,
        'rewrite_and_multiply': True,
        'equation' : '1+1'
    }]
    planner = setup(domain_al, state, ["1+1"])
    plan_next(planner)
    check_next(planner, ['rewrite_using_rule', 'multiply_exponents', 'solve_using_power_rule'], 'multiply_exponents')
    plan_next(planner)
    check_next(planner, ['rewrite_using_rule'], 'rewrite_using_rule')
    plan_next(planner)
    check_next(planner, ['solve_using_power_rule'], 'solve_using_power_rule')
    plan_next(planner)
    check_next(planner, ['done'], 'done')
    check_exhausted(planner)

    planner = setup(domain_al, state, ["1+1"])
    plan_next(planner)
    check_next(planner, ['rewrite_using_rule', 'multiply_exponents', 'solve_using_power_rule'], 'rewrite_using_rule')
    plan_next(planner)
    check_next(planner, ['multiply_exponents'], 'multiply_exponents')
    plan_next(planner)
    check_next(planner, ['solve_using_power_rule'], 'solve_using_power_rule')
    plan_next(planner)
    check_next(planner, ['done'], 'done')
    check_exhausted(planner)

    state = [{
        'id' : "1",
        'start' : True,
        'rewrite_and_multiply': False,
        'equation' : '1+1'
    }]
    planner = setup(domain_al, state, ["1+1"])
    plan_next(planner)
    check_next(planner, ['solve_using_power_rule'], 'solve_using_power_rule')
    plan_next(planner)
    check_next(planner, ['done'], 'done')
    check_exhausted(planner)
    


    
if __name__ == "__main__":
    test_basic_unord_skip()
    test_heir_unord_skip()
    test_conditional_empty_methods()
    test_optional_if()
    test_skip_if()
    test_al_domain()



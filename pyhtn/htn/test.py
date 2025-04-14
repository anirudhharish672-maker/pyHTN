from pyhtn.htn.htn_elements import Method, Operator, Task
from pyhtn.conditions.fact import Fact
from pyhtn.conditions.conditions import NOT
from pyhtn.domain.variable import V
from pprint import pprint
from pyhtn.planner.planner import HtnPlanner
from unittest.mock import MagicMock, patch


# TODO

domain = {
    #  ----  OPERATORS -------    

    "get" : Operator('get', 
        args=[V("item")], 
        # preconditions=[NOT(Fact(person='me', holding=V("anything")))],
        effects=[Fact(id='me', holding=V("item"))]
    ),

    'add_instant_coffee' : Operator('add_instant_coffee', 
        args=[V("container")], 
        # preconditions=[NOT(Fact(container=V("container"), "contents", V("anything")
        effects=[Fact(container=V("container"), solid_contents='instant_coffee')]
    ),

    'add' : Operator('add', 
        args=[V("container"), V('solid')], 
        # preconditions=[NOT(Fact(V("container"), "contents", V("anything")))],
        effects=[Fact(container=V("container"), solid_contents=V('solid'))]
    ),

    'pour' : Operator('pour', 
        args=[V("container1"), V("container2")], 
        # preconditions=[NOT(Fact(V("container"), "contents", V("anything")))],
        effects=[
            Fact(container=V("container"), liquid_contents=V('liquid'))
        ]
    ),

    'fill_with_water' : Operator('fill', 
        args=[V("container")],
        effects=[
            Fact(container=V("container"), liquid_contents=V('liquid'))
        ]
    ),

    'turn_on' : Operator('turn_on', 
        args=[V("container")],
        effects=[
            Fact(container=V("container"), is_on=True)
        ]
    ),

    'wait_for_boil' : Operator('wait_for_boil', 
        args=[V("container")],
        effects=[
            Fact(container=V("container"), is_on=True)
        ]
    ),

    #  ----  METHODS -------

    'make_coffee/0': [
        Method(
            'make_coffee',

            # Heat water is NOT primative
            subtasks=[
                Task('get', 'mug'),
                Task('add', 'mug', 'coffee'),
                Task('heat', 'water'),
                Task('pour', 'mug', 'water'),
            ],
            preconditions=[]
        ),
        Method(
            'make_coffee',

            # All of these are primative 
            subtasks=[
                Task('get', 'cup'),
                Task('add_instant_coffee', 'cup'),
                Task('heat_water_with', 'kettle'), # Non-Prim
                Task('pour', 'kettle', 'mug',),
            ],
            preconditions=[]
        )
    ],

    'heat_water_with/1': [
        Method(
            'heat_water_with', V('container'),
            subtasks=[
                Task('fill_with_water', V('container'),),
                Task('turn_on', V('container'),),
                Task('wait_for_boil', V('container'),),
            ],
            preconditions=[]
        )
    ]
}


state = [{"id" : "me", "name" : "Bob"}]

heat_water_with = domain['heat_water_with/1'][0]
fill_with_water = domain['fill_with_water']

task = Task('fill_with_water', "kettle")
op_execs = fill_with_water.get_match_executions(task, state)
print(op_execs)

task = Task('heat_water_with', "kettle")
method_execs = heat_water_with.get_match_executions(task, state)

print(method_execs)
# env = env = MagicMock()
# env.get_state.return_value = [{'id': '1', 'type': 'location', 'name': 'kitchen'}]


# HtnPlanner(
#     tasks=[],
#     domain=domain,
#     env=env,
#     enable_logging=True
# )

# pprint(domain)

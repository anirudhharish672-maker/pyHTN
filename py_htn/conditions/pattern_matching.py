from collections import defaultdict
import inspect
from py_htn.common.imports.typing import *
from py_htn.domain.variable import V
from py_plan.unification import is_variable, unify_var
from py_htn.conditions.fact import Fact
from py_htn.conditions.conditions import AND, OR, NOT, Filter


def msubst(theta: Dict, tasks: Union[Any, List, Tuple]) -> Union[Any, List, Tuple]:
    """
    Perform substitutions theta on tasks across the structure (of lists and tuples).
    """
    if not isinstance(tasks, (list, tuple)):
        s = subst(theta, tasks.head)

        result = {'name': s[0], 'args': s[1:], 'type': tasks.type}
        if tasks.type == 'operator':
            result['effects'] = tasks.effects
        return result
    else:
        return type(tasks)([msubst(theta, task) for task in tasks])


def subst(s, x):
    if isinstance(x, V):
        return subst(s, f'?{x.name}')
    if x in s:
        return s[x]
    elif isinstance(x, tuple):
        return tuple(subst(s, xi) for xi in x)
    else:
        return x


def unify(x, y, s=(), check=False):
    """
    Unify expressions x and y given a provided mapping (s).  By default s is
    (), which gets recognized and replaced with an empty dictionary. Return a
    mapping (dict) that will make x and y equal or, if this is not possible,
    then it returns None.

    >>> unify(('Value', '?a', '8'), ('Value', 'cell1', '8'), {})
    {'?a': 'cell1'}

    >>> unify(('Value', '?a', '8'), ('Value', 'cell1', '?b'), {})
    {'?a': 'cell1', '?b': '8'}
    """
    if s == ():
        s = {}

    if s is None:
        return None
    if x == y:
        return s
    if isinstance(x, V):
        return unify_var(f'?{x.name}', y, s, check)
    if isinstance(y, V):
        return unify_var(f'?{y.name}', x, s, check)

    if is_variable(x):
        return unify_var(x, y, s, check)
    if is_variable(y):
        return unify_var(y, x, s, check)

    if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
        if not x:
            return s
        return unify(x[1:], y[1:], unify(x[0], y[0], s, check), check)
    return None


def dict_to_tuple(dict_list: List[Dict]) -> List[Tuple]:
    # return [(key, d['id'], d[key]) for d in dict_list for key in d if key != 'id']
    return [(key, d['id'], d[key]) for d in dict_list for key in d]



def tuples_to_dicts(tuples: Union[List, Tuple], use_facts=False, use_and_operator=False) -> List[Dict]:
    dicts = defaultdict(dict)
    for t in tuples:
        if 'id' not in dicts[t[1]]:
            dicts[t[1]]['id'] = t[1]
        dicts[t[1]][t[0]] = t[2]
    dicts = [d if not use_facts else Fact(**d) for d in dicts.values()]
    if use_and_operator:
        dicts = AND(*dicts)
    return dicts


def fact_to_tuple(facts, variables=False):
    if isinstance(facts, Fact):
        facts = AND(facts)
    all_tuple_state = list()
    for f in generateLogics(facts):
        tuple_state = set()
        subfacts = flatten([f])
        for fact in subfacts:
            if isinstance(fact, Filter):
                vars = list(inspect.signature(fact.tmpl).parameters.keys())
                tuple_state.add((fact.tmpl, *[f'?{var}' for var in vars]))
                continue
            elif isinstance(fact, NOT):
                for cond in fact[0].conds:
                    value = f'?{cond.value.name}' if isinstance(cond.value, V) else cond.value
                    identifier = f'?{cond.identifier.name}'
                    tuple_state.add(('not', (cond.attribute, identifier, value)))
            else:
                for cond in fact.conds:
                    value = f'?{cond.value.name}' if isinstance(cond.value, V) else cond.value
                    identifier = f'?{cond.identifier.name}' if variables else cond.identifier.name
                    tuple_state.add((cond.attribute, identifier, value))
        all_tuple_state.append(tuple_state)
    return all_tuple_state

def flatten(struct):
    if not isinstance(struct, (list, tuple)) or isinstance(struct, NOT):
        return struct
    # Flatten a list if it contains another list as its element
    if isinstance(struct, list):
        flattened = []
        for item in struct:
            result = flatten(item)
            if isinstance(item, list):
                flattened.extend(result)  # Extend to flatten the list
            else:
                flattened.append(result)
        return flattened

    # Flatten a tuple if it contains another tuple as its element
    elif isinstance(struct, tuple):
        flattened = []
        for item in struct:
            result = flatten(item)
            if isinstance(item, tuple):
                flattened.extend(result)  # Extend to flatten the tuple
            else:
                flattened.append(result)
        return tuple(flattened)

    # Return the structure unchanged if it's neither a list nor a tuple
    return struct


def expandAND(*args):
    # Base case: single element
    if len(args) == 1:
        return [[arg] for arg in args[0]] if isinstance(args[0], list) else [[args[0]]]
    # Recursive case: combine each element of the first argument with the expanded result of the rest
    first, *rest = args
    rest_expanded = expandAND(*rest)
    if isinstance(first, list):
        return [[f] + r for f in first for r in rest_expanded]
    else:
        return [[first] + r for r in rest_expanded]


def expandOR(*args):
    result = []
    for arg in args:
        if isinstance(arg, list):
            result.extend(arg)
        else:
            result.append(arg)
    return result


def generateLogics(expression):
    if isinstance(expression, AND):
        return expandAND(*[generateLogics(arg) for arg in expression])
    elif isinstance(expression, OR):
        return expandOR(*[generateLogics(arg) for arg in expression])
    elif isinstance(expression, NOT):
        return [expression]
    else:
        return expression


def logics_to_dicts_and_tuples(expression: (AND, OR)) -> (list, tuple):
    exp_type = OR() if isinstance(expression, OR) else AND()
    expression = list(expression)

    for index in range(len(expression)):
        if isinstance(expression[index], AND):
            expression[index] = logics_to_dicts_and_tuples(expression[index])
        elif isinstance(expression[index], OR):
            expression[index] = logics_to_dicts_and_tuples(expression[index])
        else:
            expression[index] = {k: expression[index][k] for k in expression[index]}

    if isinstance(exp_type, AND):
        return list(expression)
    elif isinstance(exp_type, OR):
        return tuple(expression)


def generate_logics(goals: Union[list, tuple]):
    """
    Converts a list of nested lists, tuples, and dictionaries into a logical expression.
    :param goals: (List) A list containing lists (representing an AND expression),
                  tuples (representing OR expressions), and dictionaries which are converted to Facts.
    :return: AND or OR expression
    """
    terms = []
    for index in range(len(goals)):
        if isinstance(goals[index], tuple):
            terms.append(OR(*generate_logics(goals[index])))
        elif isinstance(goals[index], list):
            terms.append(AND(*generate_logics(goals[index])))
        else:
            # Element is a dictionary
            terms.append(Fact(**goals[index]))
    return terms
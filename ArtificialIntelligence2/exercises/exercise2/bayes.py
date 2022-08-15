# We implement Bayesian networks.
# For simplificity, all probability variables are Booleans.
#
# The networks are represented as Python dictionaries. Below is a sketch
# of the network for the burglary example from the lecture notes.
# It states, for example, that P(alarm | burglary, not earthquake) = 0.94.
# It follows that $P(not alarm | burglary, not earthquake) = 1-0.94.
#
# example_network = {
#     'Burglary': {'name': 'Burglary', 'parents': [], 'probabilities': {(): 0.001}},
#     'Earthquake': {'name': 'Earthquake', 'parents': [], 'probabilities': {(): 0.002}},
#     'Alarm': {
#         'name': 'Alarm',
#         'parents': ['Burglary', 'Earthquake'],
#         'probabilities': {
#             (True, True): 0.95,
#             (True, False): 0.94,
#             (False, True): 0.29,
#             (False, False): 0.001}
#         },
#     'JohnCalls': {'name': 'JohnCalls', 'parents': ['Alarm'], 'probabilities': {(True,): 0.9, (False,): 0.05}},
#     'MaryCalls': {'name': 'MaryCalls', 'parents': ['Alarm'], 'probabilities': {(True,): 0.7, (False,): 0.01}}
# }
#
# Queries consist of the network (as above), a single query variable, and an atomic event for the evidence.
# The latter is a dictionary that gives for the every evidence variable a value.
#
# Example query: query(example_network, 'Burglary', {'MaryCalls':True, 'JohnCalls':True})

def product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def P_joint(join_distribution: dict, network: dict) -> float:
    prod = 1
    for variable in join_distribution:
        prod *= P_conditional(variable, join_distribution, network)
    return prod


def P_conditional(variable: str, join_distribution: dict, network: dict) -> float:
    parents = network[variable]['parents']
    event = tuple(join_distribution[parent] for parent in parents)
    p_event = network[variable]['probabilities'][event]
    return p_event if join_distribution[variable] else 1 - p_event


def query(network, node, evidence):
    hidden_variables = set(list(network)) - set(list(evidence) + [node])
    values = [True, False]
    hidden_variables_distribution = [
        {hidden_variable: val for hidden_variable, val in zip(hidden_variables, value)}
        for value in product(values, repeat=len(hidden_variables))
    ]
    summary_pos = 0
    for hidden_variables in hidden_variables_distribution:
        join_ditribution = hidden_variables.copy()
        join_ditribution.update(evidence.copy())
        join_ditribution[node] = True
        summary_pos += P_joint(join_ditribution, network)

    summary_neg = 0
    for hidden_variables in hidden_variables_distribution:
        join_ditribution = hidden_variables.copy()
        join_ditribution.update(evidence.copy())
        join_ditribution[node] = False
        summary_neg += P_joint(join_ditribution, network)

    alpha = 1/(summary_pos + summary_neg)
    return summary_pos * alpha

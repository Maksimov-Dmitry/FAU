from math import log2
# Example query:
# dtl(
#    examples = [
#         {'Furniture': 'No', 'Nr. of rooms': '3', 'New kitchen': 'Yes', 'Acceptable': 'Yes'},
#         {'Furniture': 'Yes', 'Nr. of rooms': '3', 'New kitchen': 'No', 'Acceptable': 'No'},
#         {'Furniture': 'No', 'Nr. of rooms': '4', 'New kitchen': 'No', 'Acceptable': 'Yes'},
#         {'Furniture': 'No', 'Nr. of rooms': '3', 'New kitchen': 'No', 'Acceptable': 'No'},
#         {'Furniture': 'Yes', 'Nr. of rooms': '4', 'New kitchen': 'No', 'Acceptable': 'Yes'}
#    ],
#    attributes = {'Furniture': ['Yes', 'No'], 'Nr. of rooms': ['3', '4'], 'New kitchen': ['Yes', 'No'], 'Acceptable': ['Yes', 'No']},
#    target = 'Acceptable',
#    default = 'Yes'
# )
#
# Warning: the target attribute must not be used in the decision tree
# Warning: attributes are not necessarily binary
#
#
# Expected result:
# ('Nr. of rooms', {
#     '4': 'Yes',
#     '3': ('New kitchen', {
#         'Yes': 'Yes',
#         'No': 'No'}
#     )
#     }
# )


def entropy(probs: list) -> float:
    return sum([-p * log2(p) for p in probs])


def get_probs(outcomes: list) -> dict:
    return {i: outcomes.count(i) / len(outcomes) for i in set(outcomes)}


def get_best_attribute(examples, attributes, target) -> str:
    target_outcomes = [i[target] for i in examples]
    initial_information_gain = entropy(get_probs(target_outcomes).values())
    max_gain = -2
    best_attribute = None
    for attribute in attributes:
        attribute_outcomes = [i[attribute] for i in examples]
        attribute_probs = get_probs(attribute_outcomes)
        attribute_entropy = 0
        for outcome, outcome_prob in attribute_probs.items():
            attribute_outcomes_given_attribute = [i[target] for i in examples if i[attribute] == outcome]
            attribute_entropy += outcome_prob * entropy(get_probs(attribute_outcomes_given_attribute).values())
        if initial_information_gain - attribute_entropy > max_gain:
            max_gain = initial_information_gain - attribute_entropy
            best_attribute = attribute
    return best_attribute


def dtl_rec(examples, attributes, target, default):
    current_outcomes = list(set(i[target] for i in examples))

    if len(current_outcomes) == 1:
        return current_outcomes[0]

    if attributes is None:
        return default

    current_attributes = attributes.copy()
    best_attribute = get_best_attribute(examples, current_attributes, target)
    del current_attributes[best_attribute]
    subtree = dict()
    for attribute_outcome in attributes[best_attribute]:
        current_tree = [i for i in examples if i[best_attribute] == attribute_outcome].copy()
        if current_tree:
            subtree[attribute_outcome] = dtl_rec(current_tree, current_attributes, target, default)
    return (best_attribute, subtree)


def dtl(examples, attributes, target, default):
    current_attributes = attributes.copy()
    del current_attributes[target]
    return dtl_rec(examples, current_attributes, target, default)

import numpy as np
import network as nn
import random


SUCCESSFUL = True

# small network + example
#
# o
#  \ 0.5
#   \
#    o  b = -0.9
#   /
#  / 1
# o
#

weights_small = [np.array([[0.5,1]])]
biases_small = [np.array([-0.9])]
example_small = (np.array([0.7, 0.2]), np.array([0.1]))     # (input_, target)


# big network + example

# input layer: two neurons
# hidden layer: three neurons
# output layer: three neurons

weights_big = [np.array([[0.25216543, -0.07485984], [0.29921099, -0.00874651]]),
               np.array([[0.42306453, -0.3797564], [-0.24919561, -0.00343591], [0.42828213, 0.10834407]])]
biases_big = [np.array([0.16087441, 0.1334826]), np.array([0.07474171, -0.20047911, -0.30243764])]
example_big = (np.array([0.5, -0.5]), np.array([0.5, -0.5, 0.5]))



def check(result, should_be, thing):
    global SUCCESSFUL
    if len(result) != len(should_be):
        print(f'Wrong result for {thing}:\n{result}\nbut expected\n{should_be}')
        SUCCESSFUL = False
        return
    for i in range(len(result)):
        if np.max(np.abs(result[i] - should_be[i])) > 1e-5:
            print(f'Wrong result for {thing}:\n{result}\nbut expected\n{should_be}')
            SUCCESSFUL = False
            return


def test_apply_network():
    should_be = np.array([-0.33637554])
    result = nn.apply_network(weights_small, biases_small, example_small[0])
    check(result, should_be, 'example output for small network')

    should_be2 = np.array([0.1007614, -0.27249354, -0.1370035])
    result2 = nn.apply_network(weights_big, biases_big, example_big[0])
    check(result2, should_be2, 'example output for big network')


def test_forward():
    in_should_be = [np.array([-0.35])]
    a_should_be = [np.array([0.7, 0.2]), np.array([-0.33637554])]
    result = nn.forward(weights_small, biases_small, example_small[0])
    check(result[0], in_should_be, 'in_ for small network')
    check(result[1], a_should_be, 'a for small network')

    in_should_be2 = [np.array([0.32438705, 0.28746135]), np.array([0.10110449, -0.27955539, -0.13787047])]
    a_should_be2 = [np.array([0.5, -0.5]), np.array([0.31346831, 0.27979657]), np.array([0.1007614, -0.27249354, -0.1370035])]
    result2 = nn.forward(weights_big, biases_big, example_big[0])
    check(result2[0], in_should_be2, 'in_ for big network')
    check(result2[1], a_should_be2, 'a for big network')


def test_backward():
    in_ = [np.array([-0.35])]
    a = [np.array([0.7, 0.2]), np.array([-0.33637554])]
    should_be = [np.array([-0.3870003])]
    result = nn.backward(in_, a, weights_small, example_small[1])
    check(result, should_be, 'delta for small network')

    in_2 = [np.array([0.32438705, 0.28746135]), np.array([0.10110449, -0.27955539, -0.13787047])]
    a2 = [np.array([0.5, -0.5]), np.array([0.31346831, 0.27979657]), np.array([0.1007614, -0.27249354, -0.1370035])]
    should_be2 = [np.array([-0.43947918, 0.0752398]), np.array([-0.39518519, 0.21061349, -0.62504697])]
    result2 = nn.backward(in_2, a2, weights_big, example_big[1])
    check(result2, should_be2, 'delta for big network')


def test_update():
    deltas = [np.array([-0.3870003])]
    a = [np.array([0.7, 0.2]), np.array([-0.33637554])]
    w_should_be = [np.array([[0.52709002, 1.00774001]])]
    b_should_be = [np.array([-0.86129997])]
    result = nn.update(weights_small, biases_small, deltas, a, -0.1)
    check(result[0], w_should_be, 'updated weights for small network')
    check(result[1], b_should_be, 'updated biases for small network')

    deltas2 = [np.array([-0.43947918, 0.0752398]), np.array([-0.39518519, 0.21061349, -0.62504697])]
    a2 = [np.array([0.5, -0.5]), np.array([0.31346831, 0.27979657]), np.array([0.1007614, -0.27249354, -0.1370035])]
    w_should_be2 = [np.array([[0.27413939, -0.0968338], [0.295449, -0.00498452]]), np.array([[0.43545233, -0.36869925], [-0.25579768, -0.0093288], [0.44787537, 0.12583267]])]
    b_should_be2 = [np.array([0.20482233, 0.12595862]), np.array([0.11426023, -0.22154046, -0.23993294])]
    result2 = nn.update(weights_big, biases_big, deltas2, a2, -0.1)
    check(result2[0], w_should_be2, 'updated weights for big network')
    check(result2[1], b_should_be2, 'updated biases for big network')


def test_train_xor_gate():
    global SUCCESSFUL
    examples = []
    for i in range(10000):
        a = random.randint(0,1)
        b = random.randint(0,1)
        goal = a != b
        # True = 0.5, False = -0.5
        examples.append((np.array([a-0.5,b-0.5]), np.array([goal-0.5])))
    w, b = nn.train([2,4,2,1], examples, -0.05)
    o1 = nn.apply_network(w, b, np.array([0.5, 0.5]))[0]
    if abs(o1 + 0.5) > 0.001:
        print(f'Expected -0.5 but got {o1} (sometimes this happens randomly - see if re-running the test helps)')
        SUCCESSFUL = False
    o2 = nn.apply_network(w, b, np.array([-0.5, 0.5]))[0]
    if abs(o2 - 0.5) > 0.001:
        print(f'Expected 0.5 but got {o2} (sometimes this happens randomly - see if re-running the test helps)')
        SUCCESSFUL = False


if __name__ == '__main__':
    import sys
    for test in [test_apply_network, test_forward, test_backward, test_update, test_train_xor_gate]:
        print(f'Running {test.__name__}')
        try:
            test()
        except NotImplementedError:
            print('NotImplemented')
            SUCCESSFUL = False

    if not SUCCESSFUL:
        print('AT LEAST ONE TEST DID NOT SUCCEED')
        sys.exit(1)
    else:
        print('ALL TESTS SUCCEEDED :)')


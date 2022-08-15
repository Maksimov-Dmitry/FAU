'''
You may use this file for testing your solution.
'''
import numpy as np
from hmm import hmm_filter, hmm_predict, hmm_smooth
import sys


FAILURE = False   # Nothing has failed yet

def test(name, result, expected):
    global FAILURE
    if np.sum((result-expected)**2) > 1e-10:
        FAILURE = True
        print(f'{name} - FAILURE')
    else:
        print(f'{name} - SUCCESS')


# Example 1: Umbrellas (lecture notes)

X0 = np.array([0.5, 0.5])
T = np.array([[0.7, 0.3], [0.3, 0.7]])
O = np.array([[0.9, 0.2],    # P(umbrella | Rain)
              [0.1, 0.8]])   # P(¬umbrella | Rain)
e = [0,0]    # umbrellas appeared on days 1 and 2

test('Ex. 1, filtering ', hmm_filter(X0, T, O, e),     np.array([0.88335704, 0.11664296]))
test('Ex. 1, prediction', hmm_predict(X0, T, O, e, 4), np.array([0.56133713, 0.43866287]))
test('Ex. 1, smoothing ', hmm_smooth(X0, T, O, e, 1),  np.array([0.88335704, 0.11664296]))



# Example 2: TAs visiting bars (last week's presence problem)

X0 = np.array([1.0, 0.0])
T = np.array([[0.6, 0.4], [0.3, 0.7]])
O = np.array([[0.3, 0.8],    # P(office | Bar)
              [0.7, 0.2]])   # P(¬office | Bar)
e = [1,1]    # TA wasn't in office after days 1 and 2

test('Ex. 2, filtering ', hmm_filter(X0, T, O, e),     np.array([0.81176471, 0.18823529]))
test('Ex. 2, prediction', hmm_predict(X0, T, O, e, 3), np.array([0.54352941, 0.45647059]))
test('Ex. 2, smoothing ', hmm_smooth(X0, T, O, e, 1),  np.array([0.88235294, 0.11764706]))



# Example 3: Something random

X0 = np.array([0.2, 0.3, 0.5])
T = np.array([[0.6, 0.4, 0.0], [0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
O = np.array([[0.3, 0.5, 0.1],
              [0.4, 0.2, 0.3],
              [0.1, 0.2, 0.5],
              [0.2, 0.1, 0.1]])
e = [0,2,2,1,3,2]
test('Ex. 3, filtering ', hmm_filter(X0, T, O, e),      np.array([0.1901692,  0.49124856, 0.31858225]))
test('Ex. 3, prediction', hmm_predict(X0, T, O, e, 10), np.array([0.27472546, 0.54370972, 0.18156481]))
test('Ex. 3, smoothing ', hmm_smooth(X0, T, O, e, 3),   np.array([0.10393401, 0.35572518, 0.54034082]))



sys.exit(1 if FAILURE else 0)   # this helps us with grading


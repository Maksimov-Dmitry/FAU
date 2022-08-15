'''
Task: implement the functions hmm_* below and upload this file (with the name `hmm.py`).
Parameters:
    X0: a numpy array with the distribution at time 0.   (X0[i] = P(X_0=i))
    T:  a 2-d numpy array representing the transition model, following the conventions in the lecture.
    O:  the sensor model with O_ij = P(E=i | X=j) represented as a 2-d numpy array.
        From this, the diagonal matrices O_t can be obtained using the evidence at time t.
    e:  the observed evidence e_1,... as a python list. Following the conventions of the lecture, evidence starts at t=1, i.e. e[i] corresponds to t=i+1.
    k:  the extra time index needed for prediction and smoothing.

You may use `test.py` for testing your solution.
'''


import numpy as np    # don't use any other libraries

def normalization(P: np.ndarray) -> np.ndarray:
    return P / P.sum()

def forward(O: np.ndarray, T: np.ndarray, f: np.ndarray) -> np.ndarray:
    P = O @ T.T @ f
    return normalization(P)

def backward(O: np.ndarray, T: np.ndarray, b: np.ndarray) -> np.ndarray:
    return T @ O @ b

def hmm_filter(X0, T, O, e):
    ''' Computes P(X_t | e_{1:t}) where t := len(e) '''
    f_current = X0.reshape(-1, 1)
    for evidence in e:
        O_current = np.diag(O[evidence])
        f_current = forward(O_current, T, f_current)
    return f_current.reshape(1, -1)

def hmm_predict(X0, T, O, e, k):
    ''' Computes P(X_k | e_{1:t}) where t := len(e) and k > t.
        Note that `k` is an absolute time index here while the lecture notes use `k` as a relative offset. '''
    f_current = X0.reshape(-1, 1)
    for k_index in range(k):
        if k_index < len(e):
            O_current = np.diag(O[e[k_index]])
        else:
            O_current = np.diag(np.ones(len(X0)))
        f_current = forward(O_current, T, f_current)
    return f_current.reshape(1, -1)

def hmm_smooth(X0, T, O, e, k):
    ''' Computes P(X_k | e_{1:t}) where t := len(e) and k < t. '''
    f = hmm_filter(X0, T, O, e[:k])
    b_current = np.ones((len(X0), 1))
    for evidence in np.flip(e[k:]):
        O_current = np.diag(O[evidence])
        b_current = backward(O_current, T, b_current)
    return normalization(f * b_current.reshape(1, -1))

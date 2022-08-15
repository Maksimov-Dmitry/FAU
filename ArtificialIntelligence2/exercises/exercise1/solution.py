'''
    Your assignment is to implement the following functions,
    i.e. replace the `raise NotImplementedError()` with code that returns the right answer.

    Do not use any libraries!

    We will partly automate the evaluation so please make sure you
    that you don't change the signature of functions.
    Also please use a recent python version (at least version 3.6).
    
    You may call functions that you've implemented in other functions.
    You may also implement helper functions.

    Understanding the arguments:
        Omega:  the sample space, represented as a list
        P:      the probability function (P : Omega ⟶ float)
        ValX:   the possible values of a random variable X, represented as a list
        VarX:   a random variable (VarX : Omega ⟶ ValX), here represented as a function
        x:      a concrete value for VarX (x ∈ ValX)
        EventA: an atomic event a, represent as list of pairs [(VarA1, a1), (VarA2, a2), ...]
                representing the event a := (VarA1=a1) ∧ (VarA2=a2) ∧ …

    Example code: given Omega, P, VarX, x
        w = Omega[0]       # pick the first sample (note that the order is meaningless)
        print(P(w))        # print the probability of that sample
        if VarX(w) == x:   # compute the value of the random variable for this sample and compare it to x
            print('X = x holds for this sample')
        else:
            print('X = x doesn't hold for this sample')

    Example call:
        def isEven(n):
            if n%2 == 0:
                return 'yes'
            else:
                return 'no'
        def probfunction(n):
            return 1/6    # fair die
        print('P(isEven = yes) for a fair die:')
        print(unconditional_probability([1,2,3,4,5,6], probfunction, isEven, 'yes'))
'''


def unconditional_probability(Omega, P, VarX, x):
    ''' P(VarX = x)
        Hint: Add up P(ω) for all those ω where VarX(ω) == x by iterating over the values ω in Omega
    '''
    return sum([
        P(w)
        for w in Omega
        if VarX(w) == x
    ])


def unconditional_joint_probability(Omega, P, EventA):
    ''' P(a) '''
    P_a = 0
    for w in Omega:
        is_event = True
        for event, result in EventA:
            if event(w) != result:
                is_event = False
        if is_event:
            P_a += P(w)
    return P_a

def conditional_probability(Omega, P, VarX, x, VarY, y):
    ''' P(VarX=x|VarY=y) '''
    event = [(VarX, x), (VarY, y)]
    return unconditional_joint_probability(Omega, P, event) / unconditional_probability(Omega, P, VarY, y)

def conditional_joint_probability(Omega, P, EventA, EventB):
    ''' P(a|b) '''
    event = EventA + EventB
    return unconditional_joint_probability(Omega, P, event) / unconditional_joint_probability(Omega, P, EventB)

def probability_distribution(Omega, P, VarX, ValX):
    ''' P(VarX),
        which is defined [P(VarX = x0), P(VarX = x1), …] where ValX = [x0, x1, …]
        (return a list)
    '''
    return [unconditional_probability(Omega, P, VarX, x) for x in ValX]

def conditional_probability_distribution(Omega, P, VarX, ValX, VarY, ValY):
    ''' P(VarX|VarY)
        to be represented as a python dictionary of the form
        {(x0, y0) : P(VarX=x0|VarY=y0), …}
        for all pairs (x_i, y_j) ∈ ValX × ValY
    '''
    return {(x, y): conditional_probability(Omega, P, VarX, x, VarY, y) for x in ValX for y in ValY}

def test_event_independence(Omega, P, EventA, EventB):
    ''' P(a,b) = P(a) ⋅ P(b)
        (return a bool)
        Note: Due to rounding errors, you should only test for approximate equality (the details are up to you)
    '''
    return round(conditional_joint_probability(Omega, P, EventA, EventB), 5) == round(unconditional_joint_probability(Omega, P, EventA), 5)

def test_variable_independence(Omega, P, VarX, ValX, VarY, ValY):
    ''' P(X,Y) = P(X) ⋅ P(Y)
        (return a bool)
        Note: Due to rounding errors, you should only test for approximate equality (the details are up to you)
    '''
    return all([
        round(unconditional_probability(Omega, P, VarX, x), 5)==round(conditional_probability(Omega, P, VarX, x, VarY, y), 5)
        for x in ValX for y in ValY
    ])

def test_conditional_independence(Omega, P, EventA, EventB, EventC):
    ''' P(a,b|c) = P(a|c) ⋅ P(b|c)
        (return a bool)
        Note: Due to rounding errors, you should only test for approximate equality (the details are up to you)
    '''
    event = EventA + EventB
    common = round(conditional_joint_probability(Omega, P, event, EventC), 5)
    independence = round(conditional_joint_probability(Omega, P, EventA, EventC) * conditional_joint_probability(Omega, P, EventB, EventC), 5)
    return common == independence

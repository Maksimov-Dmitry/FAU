'''
    You can run this file to test your implementation.
    These tests are very limited and passing them is no guarantee that your code is correct.
    The testing code is rather complex and you do not have to understand it.
    Please reach out if you find any mistakes. Just make sure the most recent version still has the mistake (we plan to upload fixes).
'''


class ProbabilityModel(object):
    def __init__(self, name, Omega, P, RVs):
        self.name = name    # name of the model
        self.Omega = Omega  # sample space
        self.P = P          # probability function
        self.RVs = RVs      # random variables: {'name' : (function, values), ...}

    def __str__(self):
        rvs = list(self.RVs)
        astable = lambda l : ' | '.join([str(e).ljust(10) for e in l]) + '\n'
        s = f'Probability model: {self.name}\n'
        s += astable(['ω (∈ Ω)', f'P(ω)'] + [f'{rv}(ω)' for rv in rvs])
        for w in self.Omega:
            s += astable([w, f'{self.P(w):.5f}'] + [self.RVs[rv][0](w) for rv in rvs])
        return s

def yesnoRV(f):  # helper
    return (lambda w : 'yes' if f(w) else 'no', ['yes', 'no'])


fairCoinToss = ProbabilityModel('Fair Coin Toss', ['heads', 'tails'], lambda w : 1/2,
    {
        'isHeads': yesnoRV(lambda w : w == 'heads'),
        'isTails': yesnoRV(lambda w : w == 'tails'),
        'impossible': yesnoRV(lambda w : False),
    })

randomName = ProbabilityModel('Random Name',
    ['Fatima', 'Manuel', 'Kim', 'Ahmed', 'Luis', 'Camila', 'Muhammed', 'Maria', 'Alexander', 'Elena'],
    lambda w : 1/10,
    {
        'length' : (len, list(range(1,11))),
        'endsWithA' : yesnoRV(lambda n : n[-1] == 'a'),
        'lastLetter' : (lambda e : e[-1], list('adlmrs')),
    })

twoFairDice = ProbabilityModel('Two Fair Dice (independent)', [(a,b) for a in range(1,7) for b in range(1,7)], lambda w : 1/36,
    {
        'sum' : (sum, list(range(2,12))),
        'bothEven' : yesnoRV(lambda e : e[0]%2 == e[1]%2 == 0),
        'firstEven' : yesnoRV(lambda e : e[0]%2 == 0),
        'secondOdd' : yesnoRV(lambda e : e[1]%2 == 1),
    })

unfairDie = ProbabilityModel('Unfair Die', list(range(1,7)), lambda w : 1/2 if w == 6 else 1/10,
    {
        'isEven' : yesnoRV(lambda e : e%2 == 0),
        'isPrime' : yesnoRV(lambda e : e in [2,3,5]),
        'lessThan5' : yesnoRV(lambda e : e < 5),
    })



def test(f, cases):
    failed = False
    print(f'Testing {f.__name__}')
    try:
        for (model, args, expected, string) in cases:
            result = f(*([model.Omega, model.P] + args))
            failednow = False
            if type(expected) != type(result):
                failednow = True
                print(f'    Unexpected result type {type(result)} (expected {type(expected)})')
            elif type(expected) == float:
                if not expected - 1e-12 < result < expected + 1e-12:
                    failednow = True
                    print(f'    Failure: Expected {expected:.5f} but got {result:.5f}')
            elif type(expected) == list:
                for e,r in zip(expected, result):
                    if not e - 1e-12 < r < e + 1e-12:
                        failednow = True
                        print(f'    Failure: Expected {expected} but got {result}')
                        break
            elif type(expected) == dict:
                for k in expected:
                    if not expected[k] - 1e-12 < result[k] < expected[k] + 1e-12:
                        failednow = True
                        print(f'    Failure: Expected {expected} but got {result}')
                        break
            elif type(expected) == bool:
                if expected != result:
                    failednow = True
                    print(f'    Failure: Expected {expected} but got {result}')
            else:
                raise Exception('Unexpected error :(')
            if failednow:
                print(f'    with query {string} in the following model:\n')
                print(' '*8 + str(model).replace('\n', '\n' + ' '*8))
                print()
                failed = True
    except NotImplementedError:
        failed = True
        print('    not implemented')
    if not failed:
        print('    success')

import solution

test(solution.unconditional_probability,
        [
            (fairCoinToss, [fairCoinToss.RVs['isHeads'][0], 'yes'], 1/2, 'P(isHeads = yes)'),
            (unfairDie, [unfairDie.RVs['isEven'][0], 'yes'], 0.7, 'P(isEven = yes)'),
            (randomName, [randomName.RVs['length'][0], 5], 3/10, 'P(length = 5)'),
        ])

test(solution.unconditional_joint_probability,
        [
            (unfairDie, [[(unfairDie.RVs['isPrime'][0], 'yes'), (unfairDie.RVs['isEven'][0], 'yes')]], 1/10, 'P(isPrime=yes, isEven=yes)'),
        ])

test(solution.conditional_probability,
        [
            (unfairDie, [unfairDie.RVs['isPrime'][0], 'yes', unfairDie.RVs['isEven'][0], 'no'], 2/3, 'P(isPrime=yes|isEven=no)'),
        ])


test(solution.conditional_joint_probability,
        [
            (twoFairDice,
             [[(twoFairDice.RVs['sum'][0], 6), (twoFairDice.RVs['bothEven'][0], 'yes')],
              [(twoFairDice.RVs['firstEven'][0], 'yes')]],
             2*1/18, 'P(sum=6, bothEven=yes | firstEven=yes)'),
        ])

test(solution.probability_distribution,
        [
            (randomName, [randomName.RVs['lastLetter'][0], randomName.RVs['lastLetter'][1]], [4/10,2/10,1/10,1/10,1/10,1/10], 'P(LastLetter)')
        ])

test(solution.conditional_probability_distribution,
        [
            (unfairDie,
             [unfairDie.RVs['isEven'][0], unfairDie.RVs['isEven'][1], unfairDie.RVs['isPrime'][0], unfairDie.RVs['isPrime'][1]],
             {('yes', 'yes') : 1/3, ('yes', 'no') : 6/7, ('no', 'yes') : 2/3, ('no', 'no') : 1/7}, 'P(IsEven | IsPrime)')
        ])

test(solution.test_event_independence,
        [
            (unfairDie, [[(unfairDie.RVs['isPrime'][0], 'yes')], [(unfairDie.RVs['isEven'][0], 'no')]], False,
                'P(isPrime=yes, isEven=no) = P(isPrime=yes) * P(isEven=no)'),
            (twoFairDice, [[(twoFairDice.RVs['firstEven'][0], 'yes')], [(twoFairDice.RVs['secondOdd'][0], 'yes')]], True,
                'P(firstEven=yes, secondOdd=yes) = P(firstEven=yes) * P(secondOdd=yes)')
        ])

test(solution.test_variable_independence,
        [
            (unfairDie, [unfairDie.RVs['isPrime'][0], unfairDie.RVs['isPrime'][1], unfairDie.RVs['isEven'][0], unfairDie.RVs['isEven'][1]], False,
                'P(IsPrime, IsEven) = P(IsPrime) * P(IsEven)'),
            (twoFairDice, [twoFairDice.RVs['firstEven'][0], twoFairDice.RVs['firstEven'][1], twoFairDice.RVs['secondOdd'][0], twoFairDice.RVs['secondOdd'][1]], True,
                'P(FirstEven, SecondOdd) = P(FirstEven) * P(SecondOdd)'),
            (randomName, [randomName.RVs['lastLetter'][0], randomName.RVs['lastLetter'][1], randomName.RVs['endsWithA'][0], randomName.RVs['endsWithA'][1]], False,
                'P(LastLetter, EndsWithA) = P(LastLetter) * P(EndsWithA)'),
        ])

test(solution.test_conditional_independence,
        [
            (unfairDie, [[(unfairDie.RVs['isPrime'][0], 'yes')], [(unfairDie.RVs['isEven'][0], 'yes')], [(unfairDie.RVs['lessThan5'][0], 'yes')]], True,
                'P(isPrime=yes, isEven=yes | lessThan5=yes) = P(isPrime=yes | lessThan5=yes) * P(isEven=yes | lessThan5=yes)'),
            (unfairDie, [[(unfairDie.RVs['isPrime'][0], 'yes')], [(unfairDie.RVs['isEven'][0], 'yes')], [(unfairDie.RVs['lessThan5'][0], 'no')]], False,
                'P(isPrime=yes, isEven=yes | lessThan5=no) = P(isPrime=yes | lessThan5=no) * P(isEven=yes | lessThan5=no)'),
        ])


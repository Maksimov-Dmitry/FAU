from bayes import query


def makenode(name, parents, *probs):
    probabilities = {}
    for i in range(len(probs)//2):
        combination = tuple({'f' : False, 't' : True}[e] for e in probs[2*i])
        assert len(combination) == len(parents)
        assert combination not in probabilities
        probabilities[combination] = float(probs[2*i+1])
    assert len(probabilities) == 2**len(parents)
    return {'name' : name, 'parents' : parents, 'probabilities' : probabilities}


network1 = {
    'Asia' : makenode('Asia', [], '', 0.05),
    'Smoke' : makenode('Smoke', [], '', 0.3),
    'TBC' : makenode('TBC', ['Asia'], 't', 0.01, 'f', 0.001),
    'LC' : makenode('LC', ['Smoke'], 't', 0.2, 'f', 0.08),
    'Bron' : makenode('Bron', ['Smoke'], 't', 0.4, 'f', 0.1),
    'Xray' : makenode('Xray', ['TBC', 'LC'], 'tt', 0.98, 'tf', 0.94, 'ft', 0.92, 'ff', 0.02),
    'Dysp' : makenode('Dysp', ['TBC', 'LC', 'Bron'],
            'ttt', 0.99, 'ttf', 0.97, 'tft', 0.98, 'tff', 0.9,
            'ftt', 0.98, 'ftf', 0.92, 'fft', 0.95, 'fff', 0.07),
    }

network2 = {
    'Burglary' : makenode('Burglary', [], '', 0.001),
    'Earthquake' : makenode('Earthquake', [], '', 0.002),
    'Alarm' : makenode('Alarm', ['Burglary', 'Earthquake'], 'tt', 0.95, 'tf', 0.94, 'ft', 0.29, 'ff', 0.001),
    'JohnCalls' : makenode('JohnCalls', ['Alarm'], 't', 0.9, 'f', 0.05),
    'MaryCalls' : makenode('MaryCalls', ['Alarm'], 't', 0.7, 'f', 0.01),
    }


def testquery(network, node, evidence, expectedresult):
    estr = ', '.join([e.lower() if evidence[e] else "¬"+e.lower() for e in evidence.keys()])
    print(f'Query: P({node.lower()} | {estr})')
    result = query(network, node, evidence)
    print(f'Result: {result}')
    if abs(result - expectedresult) < 1e-10:
        print('SUCCESS!')
    else:
        print(f'I expected {expectedresult}')


if __name__ == '__main__':
    testquery(network1, 'TBC', {'Asia':False, 'Xray':True, 'Dysp':True}, 0.008321327685349975)
    print()
    testquery(network2, 'Burglary', {'MaryCalls':True, 'JohnCalls':True}, 0.2841718353643929)


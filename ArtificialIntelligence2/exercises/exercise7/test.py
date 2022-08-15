# EXAMPLE DATA

def compileExamples(attrs, examples):
    return [{x : y for x, y in zip(attrs, e)} for e in examples]

def compileAttrs(attrs, examples):
    return {attr : list({e[attr] for e in examples}) for attr in attrs}

rawattrs0 = ['Furniture', 'Nr. of rooms', 'New kitchen', 'Acceptable']
rawexamples0 = [
            ['No',        '3',            'Yes',         'Yes'],
            ['Yes',       '3',            'No',          'No'],
            ['No',        '4',            'No',          'Yes'],
            ['No',        '3',            'No',          'No'],
            ['Yes',       '4',            'No',          'Yes'],
]

rawattrs1 = ['Alt','Bar','Fri','Hun','Pat', 'Price',  'Rain','Res','Type',    'Est',     'WillWait']
rawexamples1 = [
            ["T" , "F" , "F" , "T" , "Some" , "$$$" , "F" , "T" , "French"  , "0--10"  , "T"],
            ["T" , "F" , "F" , "T" , "Full" , "$"   , "F" , "F" , "Thai"    , "30--60" , "F"],
            ["F" , "T" , "F" , "F" , "Some" , "$"   , "F" , "F" , "Burger"  , "0--10"  , "T"],
            ["T" , "F" , "T" , "T" , "Full" , "$"   , "F" , "F" , "Thai"    , "10--30" , "T"],
            ["T" , "F" , "T" , "F" , "Full" , "$$$" , "F" , "T" , "French"  , ">60"    , "F"],
            ["F" , "T" , "F" , "T" , "Some" , "$$"  , "T" , "T" , "Italian" , "0--10"  , "T"],
            ["F" , "T" , "F" , "F" , "None" , "$"   , "T" , "F" , "Burger"  , "0--10"  , "F"],
            ["F" , "F" , "F" , "T" , "Some" , "$$"  , "T" , "T" , "Thai"    , "0--10"  , "T"],
            ["F" , "T" , "T" , "F" , "Full" , "$"   , "T" , "F" , "Burger"  , ">60"    , "F"],
            ["T" , "T" , "T" , "T" , "Full" , "$$$" , "F" , "T" , "Italian" , "10--30" , "F"],
            ["F" , "F" , "F" , "F" , "None" , "$"   , "F" , "F" , "Thai"    , "0--10"  , "F"],
            ["T" , "T" , "T" , "T" , "Full" , "$"   , "F" , "F" , "Burger"  , "30--60" , "T"],
]

examples0 = compileExamples(rawattrs0, rawexamples0)
attrs0 = compileAttrs(rawattrs0, examples0)

examples1 = compileExamples(rawattrs1, rawexamples1)
attrs1 = compileAttrs(rawattrs1, examples1)


# TEST FUNCTIONS

def testCorrectness(attrs, testexamples, target, tree):
    ''' Test that the decision tree makes the right decisions for the examples '''
    correct = 0
    for e in testexamples:
        t = tree
        while t not in attrs[target]:
            t = t[1][e[t[0]]]  # descend into subtree
        if t == e[target]:
            correct += 1
    return correct

def treeSize(tree, target, leaves):
    ''' compute number of decision nodes '''
    if tree in leaves: return 0
    assert tree[0] != target
    return 1 + sum((treeSize(t, target, leaves) for t in tree[1].values()))

def treeEqual(a, b):
    ''' check for tree equality '''
    if type(a) != type(b):
        return False
    if a == b:
        return True
    if type(a) == str:
        return False
    if a[0] != b[0]:
        return False
    for key in a[1]:
        if not treeEqual(a[1][key], b[1][key]):
            return False
    return True


# MAIN

if __name__ == '__main__':
    from dtl import dtl
    tree0 = dtl(examples0, attrs0, 'Acceptable', 'Yes')
    
    correctness = testCorrectness(attrs0, examples0, 'Acceptable', tree0)
    print('You correctly classified', correctness, 'out of', len(examples0), 'examples.')

    if treeEqual(tree0, ('Nr. of rooms', {'4': 'Yes', '3': ('New kitchen', {'Yes': 'Yes', 'No': 'No'})})):
        print('Congratulations, you obtained the right tree0!')
    else:
        print('Sorry, we expected a different tree.')

    for target in ['WillWait', 'Hun', 'Type']:
        print('Target:', target)
        tree1 = dtl(examples1, attrs1, target, 'F')
        print('Tree:', tree1)
        score1 = testCorrectness(attrs1, examples1, target, tree1)
        print('You correctly classified', score1, 'out of', len(examples1), 'examples.')
        size1 = treeSize(tree1, target, attrs1[target])
        print('Your tree has', size1, 'decision nodes.')


import random

def sortSet(s):
    ss = sorted(list(s))
    sts = ','.join([str(x) for x in ss])
    return sts

class Antibody:
    antibodySet = set()
    maxfeatures = 1520
    def __init__(self, features = None, number = None):
        self.rank = 0.0
        if features is not None:
            self.features = features
        else:
            s = set()
            while True:
                while len(s) < number:
                    element = random.randint(0, Antibody.maxfeatures)
                    s.add(element)
                if sortSet(s) not in Antibody.antibodySet:
                    break
            self.features = list(s)
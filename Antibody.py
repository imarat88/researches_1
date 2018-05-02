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
            s = self.generateSet(s, number)
            self.features = list(s)
            self.feature_set = s
            self.number = number

    def generateSet(self, S = set(), number = 0):
        while True:
            s = set(S)
            while len(s) < number:
                element = random.randint(0, Antibody.maxfeatures)
                s.add(element)
            if sortSet(s) not in Antibody.antibodySet:
                return s

    def mutate(self, affinity_rank=1):

        num_mutations = min(affinity_rank, len(self.features))

        mutation_keys = random.sample(range(len(self.features)))

        for key in mutation_keys:
            self.feature_set.pop(self.features[key])

        self.feature_set = self.generateSet(self.feature_set, self.number) #mutation

        return self

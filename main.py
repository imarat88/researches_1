import pandas as pd

from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.plotting import parallel_coordinates

import random
import copy
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



antibodycombs = []

memory_antibodies = {}

antigenes = [range(1,5)]

cnum = 1519
maxnumatigenes = 5
n_neighbors = 3

bestnum = 20
iternum = 100
antibodynumber = 100

clonepercentage = 30

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

antibodySet = []


import random

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
                if element == 1520:
                    element = element
                s.add(element)
            if sortSet(s) not in Antibody.antibodySet:
                return s

    def mutate(self, affinity_rank=1):

        num_mutations = min(affinity_rank, len(self.features))

        mutation_keys = random.sample(range(len(self.features)), num_mutations)

        for key in mutation_keys:
            self.feature_set.remove(self.features[key])

        self.feature_set = self.generateSet(self.feature_set, self.number) #mutation
        self.features = list(self.feature_set)

        return self




def sortSet(s):
    ss = sorted(list(s))
    sts = ','.join([str(x) for x in ss])
    return sts

def plot3d():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100


    for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = random.randrange(n, 23, 32)
        ys = random.randrange(n, 0, 100)
        zs = random.randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def plot2d(antibody):

    dfplot = df.iloc[:, antibody.features]
    dfplot = pd.concat([dfplot, dfclass], axis=1, ignore_index=True)
    dfplot.columns = [str(x) for x in antibody.features].append("class")

    plt.figure(figsize=(15, 10))
    parallel_coordinates(dfplot, "class")
    plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Features values', fontsize=15)
    plt.legend(loc=1, prop={'size': 15}, frameon=True, shadow=True, facecolor="white", edgecolor="black")
    plt.show()



def knn(x, s):

    h = .02
    # neigh = NearestNeighbors(n_neighbors=1)
    # neigh.fit(get_training(x, s), get_labels(x))
    # neigh.kneighbors(df.iloc[i,s], n_neighbors = n_neighbors, return_distance=True)
    maxscore = 0
    for weights in ['uniform', 'distance']:
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights = weights)

        #clf.fit(get_training(x, s),get_labels(x))
        xp = get_training(x, s)
        yp = get_labels(x)
        clf.fit(xp, yp)
        score = clf.score(df.iloc[X_test,s], y_test)
        if maxscore<score:
            maxscore = score
        print(weights, score)
    return maxscore
        # if score>0.7:
        #     xp = xp.values


# neigh. predict_proba(df[i,s])

def calculateAffinity(antibody, n):
    antibody.rank = knn(X_train, antibody.features)

    return antibody.rank

def clone_antibody(antibody, clone_multiplier, num_antibodies, affinity_rank):
    num_clones = int(round(clone_multiplier * num_antibodies / float(affinity_rank)))

    return [copy.deepcopy(antibody) for i in range(num_clones)]


def get_training(x, s):
    return df.iloc[x, s]

def get_labels(y):
    xy =  np.ravel(dfclass.iloc[y])
    return xy


def generateAnibodies(antibodynumber, featurenumber):
    antibodylist = []
    for j in range(antibodynumber):
        element = Antibody(number=featurenumber)
        antibodylist.append(element)

        antibodylist[j].rank = knn(X_train, antibodylist[j].features)
    return  antibodylist


df = pd.read_csv("transBD123.csv", delimiter=";", header=0)
# print(df.columns.values)
# print(df.columns.header())
#print(df["Class"])
dfshort = df[df["Class"] == "short_acting"]
dfmedium = df[df["Class"] == "medium_acting"]
dflong = df[df["Class"] == "Long_acting"]

antibodies = []

dfclass = df.loc[:,df.columns == 'Class']

df = df.loc[:,df.columns != 'Class']

classEndoder = preprocessing.LabelEncoder()

dfclass = classEndoder.fit_transform(dfclass)
dfclass = pd.DataFrame(dfclass)
#print(df.dtypes)


x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

X = df
y = dfclass
y = dfclass.values


X_train,X_test,y_train,y_test = train_test_split(X.index,y,test_size=0.5, random_state=42)
X =  X.iloc[X_train] # return dataframe train\
X = X.values



for i in range(2,maxnumatigenes):
    first = True
    for j in range(iternum):
        antibodycombs = generateAnibodies(antibodynumber, i)
        antibodycombs.sort(key=lambda antibody:antibody.rank, reverse=True)
        antibodycomsnum = len(antibodycombs)
        affinity_rank = 0
        clones = []
        for antibody in antibodycombs[:int(antibodycomsnum*clonepercentage/100)]:
            affinity_rank = affinity_rank+1

            antibody_clones = clone_antibody(antibody, clonepercentage, antibodycomsnum, affinity_rank)

            for clone in antibody_clones:
                mutated = clone.mutate(affinity_rank) # mutated
                mutated.rank = knn(X_train, mutated.features) #affinity rates
                clones.append(mutated)

        antibodycombs.extend(clones) # combine all antibodies

        antibodycombs.sort(key=lambda antibody:antibody.rank, reverse=True) # sort to select best antibodies
        antibodycombs = antibodycombs[:antibodynumber] # cut all weak antibodies from list

    memory_antibodies[i] = antibodycombs[:bestnum]

for j in memory_antibodies[2]:
    plot2d(j)

for j in memory_antibodies[3]:
    plot3d(j)

# sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
# sss.get_n_splits(X, y)



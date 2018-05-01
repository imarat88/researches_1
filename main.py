import pandas as pd

from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import Antibody

patterns = []

antigenes = [range(1,5)]

cnum = 1520
maxnumatigenes = 5
n_neighbors = 3

iternum = 100
antibodynumber = 100

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

antibodySet = []

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

def plot2d(weights):
    x_min, x_max = xp[:, 0].min() - 1, xp[:, 0].max() + 1
    y_min, y_max = xp[:, 1].min() - 1, xp[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', alpha=0.5)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    plt.show()

def knn(i, x, s):

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

def calculateAffinity(antibody, n):
    return antibody.rank/n


            # neigh. predict_proba(df[i,s])

def get_training(x, s):
    return df.iloc[x, s]

def get_labels(y):
    xy =  np.ravel(dfclass.iloc[y])
    return xy

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
    s = set()
    for j in range(antibodynumber):
        patterns.append(Antibody(number=i))

        for k in X_test:
            patterns[j].rank = patterns[j].rank + knn(k, X_train, patterns[j].features)
        patterns[j].rank = patterns[j].rank/len(X_test) # as an indicator we take average of all scores
    nextGeneration() #TODO mutations and cloning


# sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
# sss.get_n_splits(X, y)



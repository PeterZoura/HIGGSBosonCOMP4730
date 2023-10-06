import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
data = pd.read_csv("../Dataset/higgs10k.csv", index_col=False)

#print(data.shape)
#print(data.columns)

y = data['process type']
X = data.drop(columns=['process type', 'lepton  pT', 'lepton  eta', 'lepton  phi', 'missing energy magnitude', 'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag'])
#X = data.drop(columns=['process type'])
#X = data.drop(columns=['process type', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])

'''
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=473)
param = {'n_neighbors':list(range(10, 500, 10))}
knn = KNeighborsClassifier()
kf = KFold(10, random_state=473, shuffle=True)
#knn.fit(Xtrain, ytrain).score(Xtest, ytest)
gs = GridSearchCV(knn, param, cv=kf, scoring="accuracy")
gs.fit(X, y)
print(gs.best_params_)
'''
#best neighbours is 50


'''
tested it myself as well and got 50
scli = []
for x in range(10, 500, 10):
    scores = cross_val_score(KNeighborsClassifier(x), Xtrain, ytrain, cv=10)
    scli.append((x, round(100*sum(scores)/len(scores), 3)))


#print(scli[1][1])
sclis = sorted(scli, key=lambda a : a[1])
for x in sclis:
    print(x)
'''
scores = cross_val_score(KNeighborsClassifier(50), X, y, cv=10)
print(scores.mean())

#now test for accuracy

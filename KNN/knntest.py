import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RepeatedKFold
from sklearn.metrics import get_scorer_names
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import random

random.seed()
seed = random.randrange(1000000)

data = pd.read_csv("../Dataset/higgs10k.csv", index_col=False)
#print(data.shape)
#print(data.columns)
y = data['process type']
X1 = data.drop(columns=['process type'])
X2 = data.drop(columns=['process type', 'lepton  pT', 'lepton  eta', 'lepton  phi', 'missing energy magnitude', 'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag'])
X3 = data.drop(columns=['process type', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])

X1train, X1test, ytrain, ytest = train_test_split(X1, y, test_size=2000, random_state=473)
X2train, X2test = train_test_split(X2, test_size=2000, random_state=473)
X3train, X3test = train_test_split(X3, test_size=2000, random_state=473)

param = {'n_neighbors':list(range(1, 50, 1))}
#kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)


gs = GridSearchCV(KNeighborsClassifier(),
                  param,
                  cv=10,
                  scoring={"accuracy", "roc_auc"},
                  refit="accuracy")

print("Modelling AI KNN with all attributes on 10k")
#print("Modelling AI KNN with all attributes on 100k")
gs.fit(X1train, ytrain)
print("According to accuracy: " + str(gs.best_params_) + " With score: " + str(gs.best_score_))
print("Accuracy on test data X1 is: " + str(gs.best_estimator_.score(X1test, ytest)))
print("ROC_AUC is: " + str(round(100 * roc_auc_score(ytest, gs.best_estimator_.predict(X1test)), 2)))

print("\nModelling AI KNN with 7 special attributes only on 10K")
#print("\nModelling AI KNN with 7 special attributes only on 100K")
gs.fit(X2train, ytrain)
print("According to accuracy: " + str(gs.best_params_) + " With score: " + str(gs.best_score_))
print("Accuracy on test data X2 is: " + str(gs.best_estimator_.score(X2test, ytest)))
print("ROC_AUC is: " + str(round(100 * roc_auc_score(ytest, gs.best_estimator_.predict(X2test)), 2)))

print("\nModelling AI KNN with 22 actual attributes only on 10K")
#print("\nModelling AI KNN with 22 actual attributes only on 100K")
gs.fit(X3train, ytrain)
print("According to accuracy: " + str(gs.best_params_) + " With score: " + str(gs.best_score_))
print("Accuracy on test data X2 is: " + str(gs.best_estimator_.score(X3test, ytest)))
print("ROC_AUC is: " + str(round(100 * roc_auc_score(ytest, gs.best_estimator_.predict(X3test)), 2)))

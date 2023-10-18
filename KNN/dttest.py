import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RepeatedKFold
from sklearn.metrics import get_scorer_names
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

param = {"criterion":("gini", "entropy", "log_loss"), "max_depth":(2, 3, 4, 5, 6, 7 , 8, 9, 10, 15, 20, 25, 30)}


print("Modelling AI DT with all attributes")
gs = GridSearchCV(DecisionTreeClassifier(random_state=473),
                  param,
                  cv=10,
                  scoring="accuracy")
gs.fit(X1train, ytrain)
print("According to accuracy: " + str(gs.best_params_) + " With score: " + str(gs.best_score_))
print("Accuracy on test data X1 is: " + str(gs.best_estimator_.score(X1test, ytest)))

print("\nModelling AI DT with 7 special attributes only")
gs.fit(X2train, ytrain)
print("According to accuracy: " + str(gs.best_params_) + " With score: " + str(gs.best_score_))
print("Accuracy on test data X2 is: " + str(gs.best_estimator_.score(X2test, ytest)))

print("\nModelling AI DT with 22 actual attributes only")
gs.fit(X3train, ytrain)
print("According to accuracy: " + str(gs.best_params_) + " With score: " + str(gs.best_score_))
print("Accuracy on test data X2 is: " + str(gs.best_estimator_.score(X3test, ytest)))

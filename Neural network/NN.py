import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing



data = pd.read_csv("../Dataset/higgs10k.csv", index_col=False)
data_encoded = pd.get_dummies(data, drop_first=False)

print(data_encoded)


y = (data_encoded['process type']).values
x = (data_encoded.drop(columns=['process type'])).values
#print(y)
#print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#print(x_test)


## Hyperparameters 
nn = MLPClassifier(max_iter=1000, random_state=1, early_stopping=True)
parameter_space = {
    'hidden_layer_sizes': [(100, 50, 30), (20,), (5, 2)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

## Training
clf = GridSearchCV(nn, parameter_space, n_jobs=-1, cv=10)
clf.fit(x_train, y_train)

# Get Best parameters
print('Best parameters found:\n', clf.best_params_)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
## Results
y_true, y_pred = y_test, clf.predict(x_test)
print('Results on the test set:')
print(classification_report(y_true, y_pred))
print('Accuracy: ', clf.score(x_test, y_test))


## Graphs
fig = plot_confusion_matrix(clf, x_test, y_test, display_labels=clf.classes_)
fig.figure_.suptitle("Confusion Matrix")
plt.show()



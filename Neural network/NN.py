# Import necessary libraries and modules for data manipulation, neural network modeling, and visualization.
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


# Load the dataset from the specified file path.
data = pd.read_csv("../Dataset/higgs10k.csv", index_col=False)
data_encoded = pd.get_dummies(data, drop_first=False)
print(data_encoded)


# Extract the target variable 'process type' and store it as 'y'. The rest of the data columns, excluding 'process type', are stored as 'x'. Then, the data is split into training and testing sets using an 80-20 split.
y = (data_encoded['process type']).values
x = (data_encoded.drop(columns=['process type'])).values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)


# Use the StandardScaler to normalize the data to have mean=0 and variance=1 for better performance during modeling.
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# Define the neural network model and set hyperparameters for tuning. In this case, we're using the MLPClassifier with a maximum of 1000 iterations, and early stopping enabled.
nn = MLPClassifier(max_iter=1000, random_state=1, early_stopping=True)
parameter_space = {
    'hidden_layer_sizes': [(100, 50, 30), (20,), (5, 2), (100, 50), (50, 30, 20)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.005, 0.01, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

# Measure time for training
start_time = time.time()
# Use GridSearchCV to search for the best hyperparameter by performing cross-validation on the training set.
clf = GridSearchCV(nn, parameter_space, n_jobs=-1, cv=10)
clf.fit(x_train, y_train)
# Training time
elapsed_time = time.time() - start_time
print(f"Training completed in {elapsed_time:.2f} seconds")

# Display the best hyperparameters obtained from the GridSearchCV.
print('Best parameters found:\n', clf.best_params_)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# Use the trained model to make predictions on the test set and print a classification report and accuracy score.
y_true, y_pred = y_test, clf.predict(x_test)
print('Results on the test set:')
print(classification_report(y_true, y_pred))
print('Accuracy: ', clf.score(x_test, y_test))


# Plot a confusion matrix to visualize the performance of the model on different classes in the test set.
fig = plot_confusion_matrix(clf, x_test, y_test, display_labels=clf.classes_)
fig.figure_.suptitle("Confusion Matrix")
plt.show()

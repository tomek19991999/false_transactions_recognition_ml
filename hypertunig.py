from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, classification_report, precision_recall_fscore_support
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# STEP1 - data prepare
df = pd.read_csv('dataset_false_transactions.csv')

# Make index iteration from 1
df = df.reset_index(drop=True)
n = range(1, len(df) + 1)
df.index = n

# Take transactions only with TRANSFER and CASH_OUT
X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
# take fraud transactions
Y = X['isFraud']
frauds_count = np.count_nonzero(Y)

# Delete not necessary columns and binary-encoding of labelled data in 'type'
X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int)
X = X.drop(['isFraud'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(X.head())


# Print the number of outliers
frauds_proportion = frauds_count/len(X)
print("Number of outliers in labeled data:", frauds_count)
print("Number of all transactions:", len(X))
print("Percent of outliers in labeled data: ", frauds_proportion*100)


def measure_performance(model, X_test, y_true):
    x_pred = pd.DataFrame(model.predict(X_test))
    x_pred = x_pred.replace({1: 0, -1: 1})
    matrix = confusion_matrix(x_pred, y_true)
    print(matrix)
    print(classification_report(x_pred, y_true))


#finetuning hyperparameters
n_estimators = [25, 50, 100, 150]
max_features = [1.0, 0.3,  0.5, 0.8, 0.9]
contamination_multipliers = [1, 2, 3, 5, 7, 10, 20]
bootstrap = [True]
param_grid = dict(n_estimators=n_estimators, max_features=max_features, bootstrap=bootstrap)

clf = IsolationForest(n_jobs=-1).fit(X_train)
print('Base line: default model')
measure_performance(clf, X_test, y_test)

# best parameters and performance for different contamination rates
for cont in contamination_multipliers:
    contamination = cont * frauds_proportion
    f1sc = make_scorer(f1_score, average='macro')
    clf = IsolationForest(contamination=contamination, n_estimators=n_estimators, max_features=max_features, bootstrap=False, n_jobs=-1, random_state=0)

    grid = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=f1sc)
    grid_results = grid.fit(X=X_train, y=y_train)
    best_model = grid_results.best_estimator_
    best_params = grid_results.best_params_
    print('-----------')
    print('Best params for contamination rate: ' + str(cont) + '*base fraud proportion.')
    print(best_params)
    print('Model performence: ')
    measure_performance(best_model, X_test, y_test)

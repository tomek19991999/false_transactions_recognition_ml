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


# best parameters with visualization
#Best params for contamination rate: 3*base fraud proportion.
#{'bootstrap': True, 'max_features': 0.8, 'n_estimators': 50}

clf = IsolationForest(random_state=0, contamination=3*frauds_proportion, bootstrap=True, max_features=0.8, n_estimators=50, n_jobs=-1).fit(X_train)
print('Best performence model: ')
measure_performance(clf, X_test, y_test)
X['isOutlier'] = pred = clf.predict(X)
X['isOutlier'] = X['isOutlier'].replace({1: 0, -1: 1})

outliers=X['isOutlier'].loc[X['isOutlier']==1]
outliers_index = outliers.index

X['isFraud'] = Y
frauds = Y.loc[Y==1]
fraud_index = frauds.index
map_dict = {1: 'Yes', 0: 'No'}
X['isOutlier'] = X['isOutlier'].map(map_dict)
X['isFraud'] = X['isFraud'].map(map_dict)


scatterplot = sns.scatterplot(data=X, x='amount', y='newbalanceOrig', hue='isOutlier')
plt.savefig('am(new_ori) - outlier.png')
plt.clf()

scatterplot = sns.scatterplot(data=X, x='amount', y='newbalanceOrig', hue='isFraud')
plt.savefig('am(new_ori) - fraud.png')
plt.clf()

scatterplot = sns.scatterplot(data=X, x='amount', y='oldbalanceOrg', hue='isOutlier')
plt.savefig('am(old_ori) - outlier.png')
plt.clf()

scatterplot = sns.scatterplot(data=X, x='amount', y='oldbalanceOrg', hue='isFraud')
plt.savefig('am(old_ori) - fraud.png')
plt.clf()

scatterplot = sns.scatterplot(data=X, x='amount', y='newbalanceDest', hue='isFraud')
plt.savefig('am(new_dest) - fraud.png')
plt.clf()

scatterplot = sns.scatterplot(data=X, x='amount', y='newbalanceDest', hue='isOutlier')
plt.savefig('am(new_dest) - outlier.png')
plt.clf()

scatterplot = sns.scatterplot(data=X, x='amount', y='oldbalanceDest', hue='isFraud')
plt.savefig('am(old_dest) - fraud.png')
plt.clf()

scatterplot = sns.scatterplot(data=X, x='amount', y='oldbalanceDest', hue='isOutlier')
plt.savefig('am(old_dest) - outlier.png')
plt.clf()


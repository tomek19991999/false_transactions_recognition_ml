import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb

def conf_matrix(y_test, y_predicted):
    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(15,10))
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]

    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

def check_how_many_transactions_have_no_balance_dest_before_and_after(X, Y):
    X_fraud = X.loc[Y == 1]
    X_not_fraud = X.loc[Y == 0]
    print('\nUłamek oszukańczych transakcji z \'oldbalanceDest\' = \
    \'newbalanceDest\' = 0 pomimo, że wielkość transakcji \'amount\' jest niezerowa wynosi: {:.2f} %'.\
    format((len(X_fraud.loc[(X_fraud.oldbalanceDest == 0) & \
    (X_fraud.newbalanceDest == 0) & (X_fraud.amount)]) / (1.0 * len(X_fraud)))*100))

    print('\nUłamek prawdziwych transakcji z \'oldbalanceDest\' = \
    newbalanceDest\' = 0 pomimo, że wielkość transakcji \'amount\' jest niezerowa wynosi: {:.2f} %'.\
    format((len(X_not_fraud.loc[(X_not_fraud.oldbalanceDest == 0) & \
    (X_not_fraud.newbalanceDest == 0) & (X_not_fraud.amount)]) / (1.0 * len(X_not_fraud)))*100))

def plotStrip(x, y, hue, figsize = (14, 9)):
    
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x=x, y=y, \
            hue = hue, jitter = 0.4, marker = '.', \
            size = 4, palette = colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['Not Fraud', 'Fraud'], size = 16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['Transfer', 'Cash-out'], bbox_to_anchor=(1, 1), \
               loc=2, borderaxespad=0, fontsize = 16)
    return ax

if __name__ == '__main__':

    # FLAGS
    flag_show_basic_data_visualization = True

    flag_linear_regression = True
    if flag_linear_regression:
        flag_PCA_visualization_1 = False
        flag_PCA_visualization_2 = True
        flag_PCA_visualization_3 = True


    randomState = 5
    np.random.seed(randomState)

    # STEP1 - data prepare
    df = pd.read_csv('dataset_false_transactions.csv')

    # Make index iteration from 1
    df = df.reset_index(drop=True)
    n = range(1, len(df)+1)
    df.index = n

    # Take transactions only with TRANSFER and CASH_OUT
    X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
    #take fraud transactions
    Y = X['isFraud']

    # Delete not necessary columns and binary-encoding of labelled data in 'type'
    X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)
    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.type = X.type.astype(int)

    #check_how_many_transactions_have_no_balance_dest_before_and_after(X, Y)

    # Observation: if oldbalanceDest and newbalanceDest are 0 and TRANSFER was not 0, its propably fraud. That why:
    X.loc[(X.oldbalanceDest == 0) & (X.newbalanceDest == 0) & (X.amount != 0), \
        ['oldbalanceDest', 'newbalanceDest']] = - 1
    X.loc[(X.oldbalanceOrg == 0) & (X.newbalanceOrig == 0) & (X.amount != 0), \
        ['oldbalanceOrg', 'newbalanceOrig']] = -1
    
    # STEP2 - Data visualization
    if flag_show_basic_data_visualization:
        # f(time)
        limit = len(X)
        ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])
        ax.set_ylabel('time [h]', size = 16)
        ax.set_title('Fraud and not fraud transactions in time', size = 20)

        # f(amount)
        ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))
        ax.set_ylabel('Amount', size = 16)
        ax.set_title('Distribution of real and fake transactions distributed based on the amount processed', size = 16)

    
    # STEP3 - Logistic regression with PCA
    # Shuffle data
    over = SMOTE(sampling_strategy=0.1, random_state=42)  # 10% of bigger class 
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  # 50% of smaller class after oversampling

    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    shuffled_X = X.sample(frac=1, random_state=4)
    Y = shuffled_X["isFraud"]
    del shuffled_X['isFraud']
    X, Y = pipeline.fit_resample(shuffled_X, Y)

    unique, counts = np.unique(Y, return_counts=True)
    print(np.asarray((unique, counts)).T)
    
    # Split data 8:2
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, random_state = randomState)


    # Logistic regression
    sc = StandardScaler()
    sc.fit(trainX)
    X_train_std = sc.transform(trainX)
    X_test_std = sc.transform(testX)

    # PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # Combine reduced test and training data
    X_combined_pca = np.vstack((X_train_pca, X_test_pca))
    y_combined = np.hstack((trainY, testY))

    # Training on reduced data
    log_reg = LogisticRegression(random_state=1)
    log_reg.fit(X_train_pca, trainY)
    y_pred = log_reg.predict(X_test_pca)
    accuracy = accuracy_score(testY, y_pred)
    print(f'The accuracy of the logistic regression classifier is: {accuracy * 100:.2f}%')

    if flag_PCA_visualization_1:
        plot_decision_regions(X_combined_pca, y_combined, clf=log_reg)

    # Data visualization
    if flag_PCA_visualization_2:
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=trainY, marker='o', alpha=0.5, label='Zbiór treningowy',edgecolors='g')
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=testY, marker='s', alpha=0.5, label='Zbiór testowy', edgecolors='y')
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc='upper left')
        plt.show()

    # PCA bar chart
    if flag_PCA_visualization_3:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc='upper left')
        plt.title('PCA bar chart')
        explained_variance_ratio = pca.explained_variance_ratio_
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
        plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative explained variance')
        for i, value in enumerate(explained_variance_ratio[:2]):
            plt.text(i + 0.8, value + 0.01, f"PC{i+1}: {value:.2f}", fontsize=10)
        plt.xlabel('Principal component')
        plt.ylabel('Explained variance coefficient')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    # STEP4 - XGB classificator
    weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

    clf = xgb.XGBClassifier(max_depth = 3, scale_pos_weight = weights, \
                    n_jobs = 4)
    xgb_fit = clf.fit(trainX, trainY)
    y_pred = xgb_fit.predict(testX)
    print(y_pred)
    probabilities = clf.fit(trainX, trainY).predict_proba(testX)
    print('AUPRC = {}'.format(average_precision_score(testY, probabilities[:, 1])))

    conf_matrix(testY, y_pred)
    print(classification_report(testY, y_pred))
    
    # WARNING: LONG CALCULATION on laptop
    trainSizes, trainScores, crossValScores = learning_curve(\
    xgb.XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4), trainX,\
                                            trainY, scoring = 'average_precision',\
                                            train_sizes=[0.2, 0.55, 0.775, 1.0])
    print(crossValScores)

    trainScoresMean = np.mean(trainScores, axis=1)
    trainScoresStd = np.std(trainScores, axis=1)
    crossValScoresMean = np.mean(crossValScores, axis=1)
    crossValScoresStd = np.std(crossValScores, axis=1)

    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    fig = plt.figure(figsize = (14, 9))
    plt.fill_between(trainSizes, trainScoresMean - trainScoresStd,
        trainScoresMean + trainScoresStd, alpha=0.1, color=colours[0])
    plt.fill_between(trainSizes, crossValScoresMean - crossValScoresStd,
        crossValScoresMean + crossValScoresStd, alpha=0.1, color=colours[1])
    plt.plot(trainSizes, trainScores.mean(axis = 1), 'o-', label = 'train', \
            color = colours[0])
    plt.plot(trainSizes, crossValScores.mean(axis = 1), 'o-', label = 'cross-val', \
            color = colours[1])

    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    # plt.xlim(0, 40000)
    # plt.ylim(0, 1.002)
    plt.legend(loc='best')
    for i in range(len(trainSizes)):
        plt.annotate(f'({trainSizes[i]:.0f}, {trainScoresMean[i]:.2f})', (trainSizes[i], trainScoresMean[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center')
        plt.annotate(f'({trainSizes[i]:.0f}, {crossValScoresMean[i]:.2f})', (trainSizes[i], crossValScoresMean[i]),
                    textcoords="offset points", xytext=(0, -20), ha='center')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, ['train', 'cross-val'], bbox_to_anchor=(0.8, 0.15), \
                loc=2, borderaxespad=0, fontsize = 16)
    plt.xlabel('training set size', size = 16)
    plt.ylabel('AUPRC', size = 16)
    plt.title('Learning curves indicate slightly underfit model', size = 20)
    
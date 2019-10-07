import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, RFECV
from datetime import date
from datetime import datetime
import calendar
from sklearn import metrics

# drop unnecessary columns
def drop_cols(df):
    df.drop(['date', \
         'client', \
         'industry', \
         'location', \
         'position', \
         'skillset', \
         'interview_type', \
         'cand_cur_loc', \
         'cand_job_loc', \
         'interview_loc'], \
        axis=1, \
        inplace=True)
    df.head()

def random_forest(df,X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rfs = RandomForestClassifier(n_estimators=100, random_state=5)
    rfs.fit(X_train, y_train)
    results = rfs.score(X_test, y_test)
    print('Random Forest Score: {}'.format(results))

def knn(df,X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    results = knn.score(X_test, y_test)
    print('KNN Score: {}'.format(results))

def random_forest_drop_weekdays(df,X,y):
    df_no_days.head()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rfs = RandomForestClassifier(n_estimators=100, random_state=5)
    rfs.fit(X_train, y_train)
    results = rfs.score(X_test, y_test)
    print('Random forest score without weekdays: {}'.format(results))

def knn_drop_weekdays(df,X,y):
    df_no_days.head()
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    results = knn.score(X_test, y_test)
    print('KNN score without weekdays: {}'.format(results))

def plot_roc_curve(X,y,estimator):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    estimator.fit(X_train, y_train)
    probs = estimator.predict_proba(X_test)
    probs = [p[1] for p in probs]
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    roc_auc = metrics.roc_auc_score(y_test, probs)

    plt.plot(fpr, tpr, color = 'darkorange',
            label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('data/df_cleaned.csv')
    df_no_days = df.drop(['weekdays'], axis=1)
    drop_cols(df)
    y = df.pop('obs_attend').values
    X = df.values
    X_nd = df_no_days.values
    y_nd = df_no_days.pop('obs_attend').values




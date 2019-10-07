import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import xgboost as xgb

plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-poster')


def drop_cols(df):
    df.drop(['date',
             'client',
             'industry',
             'location',
             'position',
             'skillset',
             'interview_type',
             'cand_cur_loc',
             'cand_job_loc',
             'interview_loc',
             'exp_noshow'],
            axis=1,
            inplace=True)


def make_conf_mtrx(y_pred, y_act):
    tp = np.logical_and(y_act == 1, y_pred == 1).sum()
    fp = np.logical_and(y_act == 0, y_pred == 1).sum()
    tn = np.logical_and(y_act == 0, y_pred == 0).sum()
    fn = np.logical_and(y_act == 1, y_pred == 0).sum()
    return np.array([[tp, fp], [fn, tn]])


def get_mtrx_stats(conf_mtrx):
    tp, fp, fn, tn = tuple(conf_mtrx.ravel())
    acc = (tp+tn)/(tp+fp+tn+fn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    spec = tn/(fp+tn)
    F1 = 2*prec*rec/(prec+rec)
    return acc, prec, rec, spec, F1


def print_cm_stats(y_pred, y_act):
    conf_mtrx = make_conf_mtrx(y_pred, y_act)
    print("confusion matrix:")
    print(conf_mtrx)
    acc, prec, rec, spec, F1 = get_mtrx_stats(conf_mtrx)
    print("Accuracy: {:.3f}".format(acc))
    print("Precision: {:.3f}".format(prec))
    print("Recall: {:.3f}".format(rec))
    print("Specificity: {:.3f}".format(spec))
    print("F1 Score: {:.3f}".format(F1))


def plot_roc_curve(X, y, estimator, ax):
    probs = estimator.predict_proba(X)
    probs = [p[1] for p in probs]
    fpr, tpr, threshold = metrics.roc_curve(y, probs)
    roc_auc = metrics.roc_auc_score(y, probs)

    ax.plot(fpr, tpr, color='darkorange',
            label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.legend(loc="lower right")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')


def plot_feature_importances(labels, estimator, ax):
    feat_imp = estimator.feature_importances_
    ax.bar(labels, feat_imp)
    ax.set_ylabel('Importance')
    ax.set_xlabel('Feature')
    ax.set_xticklabels(labels, rotation=90)


def calc_partial_dependence(estimator, X, feature, n=10):
    '''
    Caculates the partial dependence of X[feature] on the estimator prediction
    '''
    Xc = X.copy()
    x_feat = Xc[:, feature]
    x_feat.sort()
    x_feat = np.unique(x_feat)
    result = []
    feature_values = sample_array(x_feat, n)
    for v in feature_values:
        Xc[:, feature] = v
        y_pred = estimator.predict_proba(Xc)
        result.append(y_pred[:, 1].mean())
    return feature_values, np.array(result)


def plot_partial_dependence(estimator, X, feature, ax, n_points=10):
    '''
    Plots partial dependence of X[feature] using the provided estimator
    '''
    feat_vals, part_dep = calc_partial_dependence(estimator,
                                                  X,
                                                  feature,
                                                  n=n_points)
    ax.plot(feat_vals, part_dep)
    ax.set_ylabel('Mean Predicted Probability')


def sample_array(arr, n):
    '''
    Gets n samples evenly spaced samples (by index number) from the array arr
    '''
    step = len(arr) // (n - 1)
    if step == 0:
        return np.array(arr)
    idx = np.arange(0, len(arr), step=step, dtype=int)
    return np.array([arr[i] for i in idx])


if __name__ == '__main__':

    df = pd.read_csv('data/df_cleaned.csv')

    # Get confusion matrix, acc, prec, recall from staff predictions
    print("\n")
    print("Staff predictions")
    print_cm_stats(df['exp_noshow'], df['obs_noshow'])

    drop_cols(df)
    y = df.pop('obs_noshow')
    X = df.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

    # K-Nearest Neighbors model
    knn = KNeighborsClassifier(n_neighbors=3,
                               weights='distance',
                               algorithm='auto',
                               p=1,
                               leaf_size=20,
                               n_jobs=-1)

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("\n")
    print("K-Nearest Neighbors")
    print(knn.get_params())
    print_cm_stats(y_pred, y_test)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_roc_curve(X_test, y_test, knn, ax)
    ax.set_title('K Nearest Neighbors')
    plt.show()

    # Random Forest Classifier
    print("\n")
    print("Random Forest")
    rfs = RandomForestClassifier(n_estimators=300,
                                 criterion='gini',
                                 max_depth=None,
                                 min_samples_split=8,
                                 min_samples_leaf=1,
                                 max_features='auto',
                                 bootstrap=True,
                                 n_jobs=-1,
                                 random_state=5)
    print(rfs.get_params())
    rfs.fit(X_train, y_train)
    y_pred = rfs.predict(X_test)
    print_cm_stats(y_pred, y_test)

    # ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_roc_curve(X_test, y_test, rfs, ax)
    ax.set_title("Random Forest")
    plt.show()

    # Feature importances
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_feature_importances(df.columns, rfs, ax)
    ax.set_title("Random Forest")
    plt.show()

    # Partial Dependences (top 3 important features)
    feat_imp = rfs.feature_importances_.argsort()[:-4:-1]
    for f in feat_imp:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_partial_dependence(rfs, X_train, f, ax)
        ax.set_xlabel(df.columns[f])
        ax.set_title("Random Forest")
        plt.show()

    # Random Forest Classifier with Recursive Feature Elimination
    print("\n")
    print("Random Forest with RFE")
    rfs2 = RandomForestClassifier(n_estimators=300,
                                  criterion='gini',
                                  max_depth=None,
                                  min_samples_split=16,
                                  min_samples_leaf=1,
                                  max_features='auto',
                                  bootstrap=True,
                                  n_jobs=-1,
                                  random_state=5)
    print(rfs2.get_params())
    selector = RFE(rfs2, 3, step=1)
    selector = selector.fit(X_train, y_train)
    print('Features selected: {}'.format(selector.ranking_))
    y_pred = selector.predict(X_test)
    print_cm_stats(y_pred, y_test)

    # ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_roc_curve(X_test, y_test, selector, ax)
    ax.set_title("Random Forest with RFE")
    plt.show()

    # Feature importances
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    feat_mask = selector.ranking_ == 1
    X_train_part = X_train[:, feat_mask]
    rfs.fit(X_train_part, y_train)
    columns_part = df.columns[feat_mask]
    plot_feature_importances(columns_part, rfs, ax)
    ax.set_title("Random Forest with RFE")
    plt.show()

    # Partial Dependences
    feat_imp = rfs.feature_importances_.argsort()[::-1]
    for f in feat_imp:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_partial_dependence(rfs, X_train_part, f, ax)
        ax.set_xlabel(columns_part[f])
        ax.set_title("Random Forest with RFE")
        plt.show()

    # Gradient Boost Classifier
    print("\n")
    print("Gradient Boost")
    gb = GradientBoostingClassifier(loss='exponential',
                                    max_depth=3,
                                    learning_rate=0.01,
                                    n_estimators=100,
                                    subsample=1.0,
                                    criterion='mae',
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    max_features='auto',
                                    random_state=5)
    print(gb.get_params())
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    print_cm_stats(y_pred, y_test)

    # ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_roc_curve(X_test, y_test, gb, ax)
    ax.set_title("Gradient Boost")
    plt.show()

    # Feature importances
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_feature_importances(df.columns, gb, ax)
    ax.set_title("Gradient Boost")
    plt.show()

    # Partial Dependences (top 3 important features)
    feat_imp = gb.feature_importances_.argsort()[:-4:-1]
    for f in feat_imp:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_partial_dependence(gb, X_train, f, ax)
        ax.set_xlabel(df.columns[f])
        ax.set_title("Gradient Boost")
        plt.show()

    # Gradient Classifier with Recursive Feature Elimination
    print("\n")
    print("Gradient Boost with RFE")
    gb2 = GradientBoostingClassifier(loss='exponential',
                                     max_depth=3,
                                     learning_rate=0.01,
                                     n_estimators=100,
                                     subsample=1.0,
                                     criterion='mae',
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     max_features='auto',
                                     random_state=5)
    print(gb2.get_params())
    selector = RFE(gb2, 3, step=1)
    selector = selector.fit(X_train, y_train)
    print('Features selected: {}'.format(selector.ranking_))
    y_pred = selector.predict(X_test)
    print_cm_stats(y_pred, y_test)

    # ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_roc_curve(X_test, y_test, selector, ax)
    ax.set_title("Gradient Boost with RFE")
    plt.show()

    # Feature importances
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    feat_mask = selector.ranking_ == 1
    X_train_part = X_train[:, feat_mask]
    gb2.fit(X_train_part, y_train)
    columns_part = df.columns[feat_mask]
    plot_feature_importances(columns_part, gb2, ax)
    ax.set_title("Gradient Boost with RFE")
    plt.show()

    # Partial Dependences
    feat_imp = gb2.feature_importances_.argsort()[::-1]
    for f in feat_imp:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_partial_dependence(gb2, X_train_part, f, ax)
        ax.set_xlabel(columns_part[f])
        ax.set_title("Gradient Boost with RFE")
        plt.show()

    # Extreme Gradient Boost
    print("\n")
    print("Extreme Gradient Boost")
    DM_train = xgb.DMatrix(data=X_train, label=y_train)
    DM_test = xgb.DMatrix(data=X_test, label=y_test)

    xgbc = xgb.XGBClassifier(max_depth=2,
                             learning_rate=0.005,
                             subsample=1,
                             n_estimators=400,
                             booster='gbtree',
                             objective='binary:logistic',
                             reg_alpha=0,
                             reg_lambda=1,
                             n_jobs=-1)
    print(xgbc.get_params())
    xgbc.fit(X_train, y_train)
    y_pred = xgbc.predict(X_test)
    print_cm_stats(y_pred, y_test)

    # ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_roc_curve(X_test, y_test, xgbc, ax)
    ax.set_title("Extreme Gradient Boost")
    plt.show()

    # Feature importances
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_feature_importances(df.columns, xgbc, ax)
    ax.set_title("Extreme Gradient Boost")
    plt.show()

    # Partial Dependences (top 3 important features)
    feat_imp = gb.feature_importances_.argsort()[:-4:-1]
    for f in feat_imp:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_partial_dependence(xgbc, X_train, f, ax)
        ax.set_xlabel(df.columns[f])
        ax.set_title("Extreme Gradient Boost")
        plt.show()

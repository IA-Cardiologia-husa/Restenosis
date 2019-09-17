import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats

# from eli5.sklearn import PermutationImportance
# from eli5 import explain_weights, explain_weights_df, show_weights

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, LassoCV, lasso_path, LassoLars, LassoLarsCV, lars_path
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from sklearn.feature_selection import SelectFromModel, RFECV, RFE, VarianceThreshold, SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support,accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn import preprocessing
from sklearn.utils.multiclass import type_of_target
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# from bayes_opt import BayesianOptimization

# from skrebate import ReliefF, SURF, MultiSURF, TuRF

# from imblearn.over_sampling import SMOTENC, RandomOverSampler, ADASYN
# from imblearn.ensemble import BalancedBaggingClassifier
# from imblearn.pipeline import make_pipeline

# import multiprocessing as mp

import time


def no_sel(X,y,k=5):
    return(list(X))

def no_hyper_tun(classif,X,y):
    return classif['clf_fun']

def no_preproc(X_test, X_train):
    return X_test

def sel_feature_importances(X,y,k=5):
    clf = ExtraTreesClassifier(n_estimators=500, max_depth=2, bootstrap=True)
    clf.fit(X,y)
    importancias = pd.DataFrame()
    importancias['var'] = list(X)
    importancias['peso'] = clf.feature_importances_
    return list(importancias.sort_values(by='peso', ascending=False).reset_index(drop=True).loc[0:k,'var'])

def sel_anova(X,y,k=5):
    importancias = pd.DataFrame()
    importancias['var'] = list(X)
    importancias['peso'] = f_classif(X.fillna(X.median()),y)[0]
    return list(importancias.sort_values(by='peso', ascending=False).reset_index(drop=True).loc[0:k,'var'])
        #     SelectFromModel(clf, prefit = True, max_features=5)

def train_split(X, y, train_index, test_index, n_split, preproc, feat_sel, classif, hyper_tuning):
    X_train = X.loc[train_index,:]
    y_train = y[train_index]
    X_test  = X.loc[test_index]
    y_test = y[test_index]

    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    X_train = preproc(X_train, X_train)
    X_test = preproc(X_test, X_train)

    k = feat_sel['param_grid']['k']
    variables = feat_sel['fs_fun'](X_train, y_train, k)
#         print(variables, '\n')

    classif_opt = hyper_tuning(classif, X_train[variables], y_train)
#         print(classif_opt)

    classif_opt.fit(X_train[variables], y_train)

    y_prob_test = classif_opt.predict_proba(X_test.loc[:, variables])
    y_prob_train = classif_opt.predict_proba(X_train.loc[:, variables])

    roc_test = roc_auc_score(y_test, y_prob_test[:,1])
    roc_train = roc_auc_score(y_train, y_prob_train[:,1])
#         print('AUC test'+ str(classif['clf_name']) + ': ', roc_test)
#         print('AUC train'+ str(classif['clf_name']) + ': ', roc_train)

    print('AUC test'+ ': ', roc_test)
    print('AUC train'+ ': ', roc_train)

    df_res_test = pd.DataFrame()
    df_res_test['idx'] = test_index
    df_res_test['proba'] = y_prob_test[:,1]
    df_res_test['GT'] = list(y_test)
    df_res_test['train_test'] = 1

    df_res_train = pd.DataFrame()
    df_res_train['idx'] = train_index
    df_res_train['proba'] = y_prob_train[:,1]
    df_res_train['GT'] = list(y_train)
    df_res_train['train_test'] = 0

    df_res_test = df_res_test.append(df_res_train)

    df_res_test['split'] = n_split

    return df_res_test


def k_fold(X, y, n_folds, seed, preproc, feat_sel, classif, hyper_tuning):
    results = pd.DataFrame(columns = ['idx', 'proba', 'GT', 'train_test', 'split', 'seed'])
    skf = StratifiedKFold(n_splits = n_folds, random_state = seed, shuffle = True)
    n = 1
    for train_index, test_index in skf.split(X, y):
        res_split = train_split(X, y, train_index, test_index, n, preproc, feat_sel, classif, hyper_tuning)
        res_split['seed'] = seed
        n=n+1
        results = results.append(res_split)
#     return results
    return results


def train_split_nested(X, y, train_index, test_index, n_split, model, param_grid, inner_rkf):

    X_train = X.loc[train_index,:]
    y_train = y[train_index]
    X_test  = X.loc[test_index]
    y_test = y[test_index]

    #INNER K-FOLDS PARA OPTIMIZAR MODELO (NUMERO DE VARIABLES + HIPERPARAMETROS)
    gridcv = RandomizedSearchCV(estimator = model,
                                param_distributions= param_grid,
                                scoring = 'roc_auc', cv = inner_rkf, refit = True, return_train_score = True,
                                n_jobs = 1, random_state = 0, n_iter = 20)
    gridcv.fit(X_train, y_train)

    inner_train, inner_test = gridcv.cv_results_['mean_train_score'][gridcv.best_index_], gridcv.cv_results_['mean_test_score'][gridcv.best_index_]

    y_prob_test = gridcv.best_estimator_.predict_proba(X_test)
    y_prob_train = gridcv.best_estimator_.predict_proba(X_train)

    outer_test = roc_auc_score(y_test, y_prob_test[:,1])
    outer_train = roc_auc_score(y_train, y_prob_train[:,1])

    print('TRAIN-TEST SPLIT ', n_split)
    print('INNER TRAIN: % .3f, INNER TEST: % .3f' % (inner_train, inner_test))
    print('OUTER TRAIN: % .3f, OUTER TEST: % .3f' % (outer_train, outer_test))


    df_res_test = pd.DataFrame()
    df_res_test['idx'] = test_index
    df_res_test['proba'] = y_prob_test[:,1]
    df_res_test['GT'] = list(y_test)
    df_res_test['train_test'] = 1

    df_res_train = pd.DataFrame()
    df_res_train['idx'] = train_index
    df_res_train['proba'] = y_prob_train[:,1]
    df_res_train['GT'] = list(y_train)
    df_res_train['train_test'] = 0

    df_res_test = df_res_test.append(df_res_train)

    df_res_test['split'] = n_split

    df_res_inner = pd.DataFrame(columns = ['inner_train', 'inner_test', 'model', 'variables', 'split'])
    df_res_inner.loc[0,'inner_train'] = inner_train
    df_res_inner.loc[0, 'inner_test'] = inner_test
    df_res_inner['model'] = str(gridcv.best_estimator_)
    df_res_inner['variables'] = str(list(np.array(list(X))[gridcv.best_estimator_['fs'].get_support()]))
    df_res_inner['split'] = n_split


    return df_res_test, df_res_inner

def k_fold_nested(X, y, n_folds, seed, model, param_grid, inner_rkf):
    res_out = pd.DataFrame(columns = ['idx', 'proba', 'GT', 'train_test', 'split', 'seed'])
    res_inner = pd.DataFrame(columns = ['inner_train', 'inner_test', 'model', 'variables', 'split', 'seed'])
    skf = StratifiedKFold(n_splits = n_folds, random_state = seed, shuffle = True)
#     jobs = mp.cpu_count() // n_jobs
    n = 1
#     print(skf.get_n_splits())
    for train_index, test_index in skf.split(X, y):
#         print(train_index)
        res_split_out, res_split_in = train_split_nested(X, y, train_index,
                                                         test_index, n, model, param_grid, inner_rkf=inner_rkf)
        n=n+1
        res_split_out['seed'] = seed
        res_split_in['seed'] = seed
        res_out = res_out.append(res_split_out)
        res_inner = res_inner.append(res_split_in)

    return res_out, res_inner

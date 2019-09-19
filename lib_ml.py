import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import time

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support,accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from eli5.sklearn import PermutationImportance
from eli5 import explain_weights, explain_weights_df, show_weights

import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

### train_split_nested
## Performs a single train-test on a binary classification problem with an hyperparamter-optimized
## model through an inner k-fold iteration over the train SelectKBest
# Parameters:
# - X, y: data points and labels
# - train_index, test_index: self explanatory
# - n_split: used to save results when train_split_nested is embedded in an outer k-fold cross-validation loop
# - model: scikit-learn classification model to optimize and use
# - param_grid: scikit-learn styled hyperparameter dictionary
# - inner_rkf: scikit-learn cross_validation generator
# Returns:
# - df_res_test: dataframe with predictions for each data point.
#                Includes: index;
#                          probability of being in class 1;
#                          train (0) or test (1); n_split (for outer cross-validation purpose);
#                          seed given to the inner kf generator
# - df_res_inner: dataframe with results over the inner cv loop, used for training evaluation purposes.
#                Includes: ROC-AUC over the training set of the best model trained in the inner cv loop;
#                          ROC-AUC over the test set of the best model trained in the inner cv loop;
#                          String with construction of the best inner cv model_selection
#                          Features used for the best model (for feature selection evaluation purposes)
def train_split_nested(X, y, train_index, test_index, n_split, model, param_grid, inner_rkf):

    X_train = X.loc[train_index,:]
    y_train = y[train_index]
    X_test  = X.loc[test_index]
    y_test = y[test_index]

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
    df_res_inner['variables'] = str(list(np.array(list(X))[gridcv.best_estimator_.named_steps['fs'].get_support()]))
    df_res_inner['split'] = n_split

    return df_res_test, df_res_inner

### k_fold_nested
## function to perform a nested cv stratified k-fold iteration in a binary classification problem and save predictions.
## Model and hyperparameters are optimized on an inner loop.
## Parameters: see train_split_nested
## Returns: dataframe with concatenated results of each train_test split of the outer k-fold.
def k_fold_nested(X, y, n_folds, seed, model, param_grid, inner_rkf):
    print( '** STARTING K-FOLD WITH SEED % d **' % (seed))
    res_out = pd.DataFrame(columns = ['idx', 'proba', 'GT', 'train_test', 'split', 'seed'])
    res_inner = pd.DataFrame(columns = ['inner_train', 'inner_test', 'model', 'variables', 'split', 'seed'])
    skf = StratifiedKFold(n_splits = n_folds, random_state = seed, shuffle = True)
    n = 1
    for train_index, test_index in skf.split(X, y):
        res_split_out, res_split_in = train_split_nested(X, y, train_index,
                                                         test_index, n, model, param_grid, inner_rkf=inner_rkf)
        n=n+1
        res_split_out['seed'] = seed
        res_split_in['seed'] = seed
        res_out = res_out.append(res_split_out)
        res_inner = res_inner.append(res_split_in)

    print( '** SEED % d FINISHED**' % (seed))

    return res_out, res_inner


import os
import re
import sys
import pandas as pd
import numpy as np

#to plot the data
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import class_weight

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split

from fitter import Fitter, get_common_distributions, get_distributions

from collections import Counter
import missingno as mn
import random
from copy import deepcopy

from sklearn import ensemble
from sklearn import tree
from sklearn import kernel_ridge

from sklearn.preprocessing import scale

from sklearn import linear_model
from sklearn import decomposition
from sklearn import svm
import catboost
import lightgbm

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn import model_selection

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

from scipy.stats import pearsonr
from dtaidistance import dtw, dtw_ndim
from kneed import DataGenerator, KneeLocator
from  sklearn.feature_selection import VarianceThreshold

from statsmodels.tsa.stattools import adfuller
from scipy.fft import fft, ifft
from IPython.display import display, Markdown, Image

%matplotlib inline

import warnings
warnings.simplefilter("ignore")



def regressions_helper():
    ## place holder for regresson models
    #linear model
        #SGDRegressor
        #OrthogonalMatchingPursuit
        # BayesianRidge
        #LinearRegression
        #LassoLars
        #Ridge
        #LogisticRegression
        #LinearRegression
        #ARDRegression
        #ElasticNet
        #Lasso
        #SVR
        #PassiveAggressiveRegressor

    # ensemble
        #GradientBoostingRegressor
        #RandomForestRegressor

    # tree
        # DecisionTreeRegressor

    # cross decomposition
        #cross_decomposition. PLSRegression

    # kernel_ridge.KernelRidge
        # kernel_ridge.KernelRidge

    # catboost
        #catboost.CatBoostRegressor

    linear_reg = linear_model.LinearRegression(fit_intercept=True,  n_jobs=None, positive=False)
    #n_jobs: 1 cpu, -1 all cpus, positive: if true then coeff will be >0 (only for dense arrays). #normalize can be set to true. fit_intercept when True: will be normalized
    #  fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=False

    lassor_lars = linear_model.LassoLars(alpha=1.0,  fit_intercept=True, verbose=False, normalize='deprecated', precompute='auto', max_iter=500)
    # Least Angle Regression: lars.  least-angle regression (LARS) is an algorithm for fitting linear regression models to high-dimensional data
    # alpha: Constant that multiplies the penalty term. Defaults to 1.0. alpha = 0 is equivalent to an ordinary least square
    # alpha=1.0, *, fit_intercept=True, verbose=False, normalize='deprecated', precompute='auto',
    #max_iter=500, eps=2.220446049250313e-16, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None

    ridge = linear_model.Ridge(alpha = 0.5)
    # alpha: ridge regulation. the lambda
    # solver{'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'}, default='auto'
    # alpha=1.0, *, fit_intercept=True, normalize='deprecated', copy_X=True, max_iter=None, tol=0.001, solver='auto', positive=False, random_state=None

    logit = linear_model.LogisticRegression(penalty='elasticnet', l1_ratio = 0.5)
    # penalty: 'l1', 'l2', 'elasticnet'  l1_ratio: only when penalty='elasticnet'
    # penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
    #random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None


    lr = linear_model.LinearRegression(fit_intercept=True, positive=False)
    # positive: make coef to be positive

    baysesain_ard = linear_model.ARDRegression(fit_intercept=True)
    #ARD Automatic Relevance Determination Regression
    # Bayesian ARD. Fit the weights of a regression model, using an ARD prior. ard Automatic Relevance Determination Regression
    # n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False,
    #threshold_lambda=10000.0, fit_intercept=True, normalize='deprecated', copy_X=True, verbose=False

    bay_ridge = linear_model.BayesianRidge()
    # Bayesian ridge regression

    elastic = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
    # ElasticNet

    lasso = linear_model.Lasso(alpha=1.0,  fit_intercept=True)
    # Lasso

    svr = svm.SVR(kernel='rbf', degree=3, gamma='scale')
    # kernel{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}

    twee = linear_model.TweedieRegressor(alpha=1.0, fit_intercept=True, link='auto', max_iter=500)
    # Generalized Linear Model with a Tweedie distribution
    # power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=0.0001, warm_start=False, verbose=0
    # power is for distribution. 0 : normal

    omp = linear_model.OrthogonalMatchingPursuit(fit_intercept=True)
    # Orthogonal Matching Pursuit model (OMP)
    # n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize='deprecated', precompute='auto'
    # n_nonzero_coefs:

    sgd = linear_model.SGDRegressor(loss='squared_error',  penalty='l2', alpha=0.0001, l1_ratio=0.15)
    # loss='squared_error', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000,
    # tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False

    pag = linear_model.PassiveAggressiveRegressor(max_iter = 100)
    # Passive Aggressive Regressor.
    # *, C=1.0, fit_intercept=True, max_iter=1000, tol=0.001, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0,
    #loss='epsilon_insensitive', epsilon=0.1, random_state=None, warm_start=False, average=False

    en_gbr = ensemble.GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100)
    #Gradient Boosting for regression.
    #  loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
    # min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None,
    # max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0

    en_rf = ensemble.RandomForestRegressor(n_estimators=100,  criterion='squared_error')
    # ensemble
    # n_estimators=100, *, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0,
    # max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None

    tree_reg = tree.DecisionTreeRegressor(criterion='squared_error', splitter='best')
    # A decision tree regressor.

    pls = cross_decomposition.PLSRegression()
    # Partial least squares regression PLSRegression, with multivariate response, a.k.a. PLS2

    kernel_ridge = kernel_ridge.KernelRidge(kernel='linear', gamma=None, degree=3)
    # Kernel ridge regression.
    #alpha=1, *, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None
    # alpha: regulation strength

    cat_reg = catboost.CatBoostRegressor(loss_function="RMSE" )
    # CatBoostRegressor
    #https://catboost.ai/en/docs/concepts/parameter-tuning

    lgbm_reg = lightgbm.LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100)
    # LGBMRegressor
    # boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None,
    # class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
    # reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=None, importance_type='split', **kwargs



    def prepare(random_state = 123, use_test_set = True):
        tmp = DF_TRAIN.copy()
        X = tmp[FEATS]
        y = tmp.pop("rul")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_state)

        if use_test_set:
            X_test = DF_TEST.copy()
            X_test = X_test.drop('cycle', axis=1).groupby('id').last().copy().reset_index()# get last row of each engine
            y_test = X_test['remaining_rul'].copy()

        return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare()

def pipeline_gridsearch():
    """
        doc:
        grid search in pipelines
    """
    names = [  "linear_reg", #1
    "lassor_lars", #2
    "ridge", #3
    "logit", #4
    "ard", #5
    "bayesian_ridge",  #6
    "elastic", #7
    "lasso", #8
    "svr", #9 very slow
    "twee", #10
    "omp", #11
    "sgd", #12
    "pag", #13
    "en_gbr", #14
    "en_rf", #15
    "tree_reg", #16
    #"pls", #17
    "kernel_ridge", #17
    "cat_reg", #18
    "lgbm_reg", #19
    ]

    regressors = [
     linear_model.LinearRegression(), #1
     linear_model.LassoLars(),  #2
     linear_model.Ridge(), #3
     linear_model.LogisticRegression(),  #4
     linear_model.ARDRegression(), #5
     linear_model.BayesianRidge(), #6
     linear_model.ElasticNet(), #7
     linear_model.Lasso(), #8
     svm.SVR(), #9 very slow
     linear_model.TweedieRegressor(), #10
     linear_model.OrthogonalMatchingPursuit(), #11
     linear_model.SGDRegressor(), #12
     linear_model.PassiveAggressiveRegressor(), #13
     ensemble.GradientBoostingRegressor(), #14
     ensemble.RandomForestRegressor(), #15 very slow
     tree.DecisionTreeRegressor(), #16
     #cross_decomposition.PLSRegression(), #
     kernel_ridge.KernelRidge(), #17
     catboost.CatBoostRegressor(), #8
     lightgbm.LGBMRegressor()  #19
    ]

    parameters = [
                  {'fit_intercept': [True], "positive":[True]},  #1
                  {'fit_intercept': [True,False], "positive": [True, False], "alpha": [0, 1.0]},  #alpha: 0  is equal to ols  #2
                  {'fit_intercept': [True,False], "positive": [True, False], "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "lbfgs"]},  #alpha: 0  is equal to ols  #3
                  {"penalty": ["l2", "l1", "elasticnet"],  'fit_intercept': [True,False],  "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "lbfgs"]},  #alpha: 0  is equal to ols   #4
                  {'fit_intercept': [True,False], "n_iter":[200, 500] },   #5
                  {'fit_intercept': [True,False],   "n_iter":[200, 300, 400]},   #6
                  {'alpha':[0.2, 0.5, 1.0], "l1_ratio": [0.1, 0.5, 0.8],  "max_iter":[1000, 400]},   #7
                  {'alpha':[0.2, 0.5, 1.0],  "max_iter":[1000, 400]},   #8
                  {'kernel':['linear'],  "gamma":["auto", "scale"]},   #9 linear’, ‘poly’, ‘rbf’, ‘sigmoid
                  {'power':[0, 3] },   #10
                  {'n_nonzero_coefs':[None, 5, 14] }, #11
                  {'loss':["squared_error","huber","epsilon_insensitive"], "penalty": ["l2","l1","elasticnet"], "l1_ratio":[0.15, 0.5]},  #12
                  {'C':[1.0, 0.5], "early_stopping":[True, False]}, #13
                  {'loss':['squared_error', 'absolute_error',  'huber'] }, #14
                  {'criterion':['squared_error', 'absolute_error', 'poisson'], "n_estimators":[100, 200] }, #15
                  {'criterion':['squared_error', 'absolute_error',  'poisson'] },  #16
                  {'alpha':[1.0, 0.8], "kernel": ["linear"]},#17
                  {'loss_function':["RMSE"], 'iterations': [100, 150, 200],#18
                    'learning_rate': [0.03, 0.1],
                    'depth': [2, 4, 6, 8],
                    'l2_leaf_reg': [0.2, 0.5, 1, 3]},
                  {'learning_rate':[0.005, 0.10], "feature_fraction": [0.5, 0.9, 1.0], "boosting_type": ["gbdt", "dart"] }, #19
                   ]

    processed = [ ]
    history = {}
    for  name, classifier, params in  zip(names, regressors, parameters):
        if name   in ["svr", "en_rf"]: ## not to proceed. too slow
            continue
        if name in processed:
            continue
        if name in history:
            continue
        pipe=Pipeline([(name, classifier)])
        params = {name+"__"+k: v for k, v in params.items()} # by convention 约定俗成
        gs_clf = GridSearchCV(pipe, param_grid=params, n_jobs=-1, scoring="r2" )
        clf = gs_clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        history[name] = {}
        history[name]["score"] = score
        history[name]["regressor"] = gs_clf

        processed.append(name)
    return history, processed




best_scores = {}
best_estimators = {}
for k,v in history.items():
    best_scores[k] = history[k]["regressor"].best_score_
    best_estimators[k] = history[k]["regressor"].best_estimator_.get_params()

best = {k: v if v>0 else 0 for k,v in best_scores.items()}

plt.figure(figsize=(15,6))

plt.plot(best.keys(), best.values())
plt.xticks(rotation = 45, fontsize=12)
plt.axhline(y = 0.5, color = 'r', linestyle = '-')
plt.show()
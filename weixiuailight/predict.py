import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn.utils import shuffle
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import (mean_squared_error,mean_absolute_error, classification_report, confusion_matrix,\
    accuracy_score,f1_score,  roc_curve, roc_auc_score,r2_score, precision_score, recall_score)

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)


FEATURES = ['sensor2',
 'sensor3',
 'sensor4',
 'sensor7',
 'sensor8',
 'sensor9',
 'sensor11',
 'sensor12',
 'sensor13',
 'sensor14',
 'sensor15',
 'sensor17',
 'sensor20',
 'sensor21']

def predict():
    """
    Return data for inference.
    """

    test_file = "../data/test.csv"
    data_test = pd.read_csv(test_file)
    data_test.rename(columns={"remaining_rul": "rul"})
    X_test = data_test[FEATURES]
    y_train = data_test['rul'].values

    # #############################################################################
    # Load model
    print("Loading model from: {}".format(MODEL_PATH))
    clf = joblib.load(MODEL_PATH)

    # #############################################################################
    # Run inference
    print("Scoring observations...")
    y_pred = clf.predict(X_test)
    print(y_pred)

if __name__ == '__main__':
    predict()
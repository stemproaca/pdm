import os
import json
from joblib import dump, load
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

# #############################################################################
# Load directory paths for persisting model and metadata

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

def train():

    # Load, read and normalize training data
    training_file = "../data/train.csv"
    data_train = pd.read_csv(training_file)
    data_train["rul"] = data_train.groupby(["Flag","id"])["cycle"].transform("max")-data_train["cycle"]

    X_train, X_test, y_train, y_test = train_test_split(data_train[FEATURES], data_train['rul'].values, test_size=0.4)
    #X_train = data_train[FEATURES]
    #y_train = data_train['rul'].values

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)

    reg = LGBMRegressor(boosting_type='gbdt', objective='regression', num_leaves=1200,
                                    learning_rate=0.17, n_estimators=10, max_depth=4,
                                    metric='rmse', bagging_fraction=0.8, feature_fraction=0.8, reg_lambda=0.9)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_absolute_error( y_test, predictions)

    ############################################################################# save model
    # Save model
    print("Serializing model to: {}".format(MODEL_PATH))
    dump(reg, MODEL_PATH)

    ############################################################################# dump a meta data file
    metadata = {
    "test_mean_square_error": mse,
    "test_r2": r2
    }

    # #############################################################################
    # Serialize model and metadata

    print("Serializing metadata to: {}".format(METADATA_PATH))
    with open(METADATA_PATH, 'w') as outfile:
        json.dump(metadata, outfile)

if __name__ == '__main__':
    train()
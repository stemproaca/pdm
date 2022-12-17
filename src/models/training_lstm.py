import os
import re
import sys
import pickle
from copy import deepcopy

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, TimeDistributed
from tensorflow.keras.models import load_model

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CURRENT_FILE_PATH_PARENT = os.path.dirname(CURRENT_FILE_PATH)
PRPOJECT_PATH = os.path.dirname(CURRENT_FILE_PATH_PARENT)
FEATUREFILE = CURRENT_FILE_PATH_PARENT+"/features/FeatureCSV.csv"

sys.path.append(CURRENT_FILE_PATH_PARENT)
from data.data_utility import DataUtility
from features.feature_selection_eda import SelectingEDAFeatures
from serving.excel_reporting import ExcelUtility

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

class RulLSTM(DataUtility):
    def __init__(self, sequence_length = 6,
                 n_samples=4,
                 alpha=0.4,
                 n_splits=1,
                 train_size=0.80,
                 label="rul",
                 random_state=123,
                 add_op=True,
                 to_ewm=True,
                 to_scale=True,
                 to_clip=True,
                 clipper_upper=150,
                 best_model_file_prefix = None,
                 flags="FD001"):

        DataUtility.__init__(self)

        self.df_train, self.df_test, self.df_result = self.load_data_by_flags(flags=flags)
        self.sequence_length = sequence_length
        if not os.path.exists(FEATUREFILE):
            eda = SelectingEDAFeatures()
            eda.eda_features()
        df_ = pd.read_csv(FEATUREFILE)
        self.sensors = df_["Feature"].values
        self.sensors = [c for c in self.sensors if re.search(r"sensor", c)]

        self.n_splits = n_splits
        self.n_samples = n_samples
        self.alpha = alpha
        self.label = label
        self.random_state = random_state
        self.train_size = train_size
        self.add_op=add_op
        self.to_ewm=to_ewm
        self.to_scale=to_scale
        self.to_clip=to_clip
        self.clipper_upper=clipper_upper

        self.gss = GroupShuffleSplit(n_splits=self.n_splits, train_size=self.train_size, random_state=self.random_state)
        if not best_model_file_prefix:
            best_model_file_prefix = "rul_lstm_n"
        self.best_model_file_prefix = best_model_file_prefix

    def add_operating_condition(self, df):
        df_op_cond = df.copy()
        df_op_cond['op1'] = df_op_cond['op1'].round()
        df_op_cond['op2'] = df_op_cond['op2'].round(decimals=2)

        # converting settings to string and concatanating makes the operating condition into a categorical variable
        df_op_cond['op_cond'] = df_op_cond['op1'].astype(str) + '_' + \
                            df_op_cond['op2'].astype(str) + '_' + \
                            df_op_cond['op3'].astype(str)

        return df_op_cond

    def scaling(self, df_train=pd.DataFrame(), df_test=pd.DataFrame(), sensors=None, add_op = True):
        # apply operating condition specific scaling

        if df_train.empty:
            df_train = self.df_train.copy()
        if df_test.empty:
            df_test = self.df_test.copy()

        if not sensors:
            sensors = self.sensors

        scaler = StandardScaler()

        if add_op:
            df_train = self.add_operating_condition(df = df_train)
            df_test = self.add_operating_condition(df = df_test)
            for condition in df_train['op_cond'].unique():
                scaler.fit(df_train.loc[df_train['op_cond']==condition, sensors])
                df_train.loc[df_train['op_cond']==condition, sensors] = \
                    scaler.transform(df_train.loc[df_train['op_cond']==condition, sensors])
                df_test.loc[df_test['op_cond']==condition, sensors] = \
                    scaler.transform(df_test.loc[df_test['op_cond']==condition, sensors])
        else:
            scaler.fit(df_train[sensors])
            df_train[sensors] = scaler.transform(df_train[sensors])
            df_test[sensors] = scaler.transform(df_test[sensors])

        return df_train, df_test

    def create_mask(self, data, samples):
        #drop first n_samples of each id to reduce filter delay
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    def ewm_data(self, df, sensors=None, n_samples=None, alpha=None ):
        # ewm(span : not set. maybe set to 4)
        # n_samples: beginning n_samples rows for each engine id
        if not sensors:
            sensors = self.sensors
        if not n_samples:
            n_samples = self.n_samples
        if not alpha:
            alpha = self.alpha

        df_smoothen = df.copy()
        # first, take the exponential weighted mean
        df_smoothen[sensors] = df_smoothen.groupby('id')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean())

        mask = df_smoothen.groupby('id')['id'].transform(self.create_mask, samples=n_samples).astype(bool)
        df_smoothen = df_smoothen[mask]
        return df_smoothen

    def gen_train_data(self, df, sensors, sequence_length):
        data_ = df[sensors].values
        num_elements = data_.shape[0]

        # -1 and +1 because of Python indexing
        for start, stop in zip(range(0, num_elements-(sequence_length-1)), \
                               range(sequence_length, num_elements+1)):
            yield data_[start:stop, :]


    def gen_data_wrapper(self, df, sequence_length=None, sensors=None, ids=None):
        if not sequence_length:
            sequence_length = self.sequence_length
        if not sensors:
            sensors = self.sensors

        data_gen = (list(self.gen_train_data(df[df['id']==id], sensors, sequence_length))
                   for id in ids)
        data_array = np.concatenate(list(data_gen)).astype(np.float32)

        return data_array

    def gen_labels(self, df,sequence_length, label):
        #data_np = df[[self.label]].values
        data_np = df[label].values
        num_elements = data_np.shape[0]

        # -1 because I want to predict the rul of that last row in the sequence, not the next row
        return  data_np[sequence_length-1:num_elements, :]

    def gen_label_wrapper(self, df, sequence_length=None, label=None, ids=None):
        if not sequence_length:
            sequence_length = self.sequence_length
        if not label:
            label = self.label
        if isinstance(label, str):
            label = [label]


        label_gen = [self.gen_labels(df[df['id']==id], sequence_length, label)
                    for id in ids]
        label_array = np.concatenate(label_gen).astype(np.float32)
        return label_array



    def gen_test_data(self, df, sequence_length, mask_value):
        if df.shape[0] < sequence_length:
            data_np = np.full(shape=(sequence_length, len(self.sensors)), fill_value=mask_value) # pad
            idx = data_np.shape[0] - df.shape[0]
            data_np[idx:,:] = df[self.sensors].values  # fill with available data
        else:
            data_np = df[self.sensors].values

        # specifically yield the last possible sequence
        stop = num_elements = data_np.shape[0]
        start = stop - sequence_length
        for i in list(range(1)):
            yield data_np[start:stop, :]

    def generic_split(self, X, y, groups, print_groups=False):
        # groups: groups=X_train_interim['id'].unique()
        # groups is ids
        for idx_train, idx_val in self.gss.split(X, y, groups=groups):
            if print_groups:
                print('train_split_engines', X.iloc[idx_train]['id'].unique())
                print('validate_split_engines', X.iloc[idx_val]['id'].unique(), '\n')

            X_train_split = X.iloc[idx_train].copy()
            y_train_split = y.iloc[idx_train].copy()
            X_val_split = X.iloc[idx_val].copy()
            y_val_split = y.iloc[idx_val].copy()
        return X_train_split, y_train_split, X_val_split, y_val_split

    def padding(self):
        ### information: how padding works
        # padding example
        a = np.full(shape=(5,3), fill_value=-99.)  # desired sequence length
        b = np.full(shape=(2,3), fill_value=0.)  # available sequence length
        idx = a.shape[0] - b.shape[0]  # equals to 3
        a[idx:,:] = b


    def prepare_data(self):
        train, test = self.df_train, self.df_test

        if self.to_scale:
            train = self.add_operating_condition(train)
            test = self.add_operating_condition(test)
        if self.to_ewm:
            train = self.ewm_data(train)
            test = self.ewm_data(test)

        if self.to_clip:
            train[self.label].clip(upper=125, inplace=True)

        for train_unit, val_unit in self.gss.split(train['id'].unique(), groups=train['id'].unique()):
            train_unit = train['id'].unique()[train_unit]  # gss returns indexes and index starts at 1
            val_unit = train['id'].unique()[val_unit]

            train_split_array = self.gen_data_wrapper(train, None, None, train_unit)
            train_split_label = self.gen_label_wrapper(train, None, None, train_unit)

            val_split_array = self.gen_data_wrapper(train,None, None,  val_unit)
            val_split_label = self.gen_label_wrapper(train,  None, None, val_unit)

        return train_split_array, train_split_label, val_split_array, val_split_label

    def build_model(self):
        train_split_array, train_split_label, val_split_array, val_split_label = self.prepare_data()
        model = Sequential()
        model.add(Masking(mask_value=-99., input_shape=(self.sequence_length, train_split_array.shape[2])))
        model.add(LSTM(32, activation='tanh'))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.save_weights('simple_lstm_weights.h5')

        model.compile(loss='mean_squared_error', optimizer='adam')  # the model is recompiled to reset the optimizer
        model.load_weights('simple_lstm_weights.h5')  # weights are reloaded to ensure reproducible results


        history = model.fit(train_split_array, train_split_label,
                            validation_data=(val_split_array, val_split_label),
                            epochs=5,
                           batch_size=32)
        return history, model

    def plot_loss(self, fit_history):
        # plot history
        plt.figure(figsize=(13,5))
        plt.plot(range(1, len(fit_history.history['loss'])+1), fit_history.history['loss'], label='train')
        plt.plot(range(1, len(fit_history.history['val_loss'])+1), fit_history.history['val_loss'], label='validate')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def predict_evaluate(self, model_file=None, param_file=None):
        # predict and evaluate
        train_split_array, train_split_label, val_split_array, val_split_label = self.prepare_data()

        if not model_file:
            _, model = self.build_model()
        else:
            model = load_model(model_file)

        y_hat_train = model.predict(train_split_array)
        self.evaluate(train_split_label, y_hat_train, 'train')

        y_hat_test = model.predict(val_split_array)
        self.evaluate(val_split_label, y_hat_test)

    def evaluate(self, y_true, y_hat, label='test'):
        mse = mean_squared_error(y_true, y_hat)
        rmse = np.sqrt(mse)
        variance = r2_score(y_true, y_hat)
        print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))


    def predict(self, prod_data, model_file = None, param_file=None):
        if not model_file:
            _, model = self.build_model()
        else:
            model = load_model(model_file)

        if not param_file:
            batch_size = self.batch_size
            sequence_length = self.sequence_length
        else:
            df_ = pd.read_csv(param_file)
            batch_size, sequence_length = df_.loc[0]["batch_size"], df_.loc[0]["sequence_length"]

        if self.to_scale:
            prod_data = self.add_operating_condition(prod_data)
        if self.to_ewm:
            prod_data = self.ewm_data(prod_data)

        prod_data_gen = (list(self.gen_test_data(prod_data[prod_data['id']==id], sequence_length,  -99.))
                   for id in prod_data['id'].unique())
        prod_array = np.concatenate(list(prod_data_gen)).astype(np.float32)

        results = model.predict(prod_array)

        return results

    # input_shape = (sequence_length, train_array.shape[2])
    def build_model_full(self, input_shape, nodes_per_layer, dropout, activation, weights_file):
        model = Sequential()
        model.add(Masking(mask_value=-99., input_shape=input_shape))
        if len(nodes_per_layer) <= 1:
            model.add(LSTM(nodes_per_layer[0], activation=activation))
            model.add(Dropout(dropout))
        else:
            model.add(LSTM(nodes_per_layer[0], activation=activation, return_sequences=True))
            model.add(Dropout(dropout))
            model.add(LSTM(nodes_per_layer[1], activation=activation))
            model.add(Dropout(dropout))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.save_weights(weights_file)

        return model

    def prep_data(self, train, test, drop_sensors, remaining_sensors, alpha):
        X_train_interim = self.add_operating_condition(train.drop(drop_sensors, axis=1))
        X_test_interim = self.add_operating_condition(test.drop(drop_sensors, axis=1))


        #scaling(self, df_train=pd.DataFrame(), df_test=pd.DataFrame(), sensors=None, add_op = True)
        X_train_interim, X_test_interim = self.scaling(X_train_interim, X_test_interim, remaining_sensors, add_op=False)

        # ewm_data(self, df, sensors=None, n_samples=None, alpha=None )
        X_train_interim = self.ewm_data(X_train_interim, remaining_sensors, 0, alpha)
        X_test_interim = self.ewm_data(X_test_interim, remaining_sensors, 0, alpha)

        return X_train_interim, X_test_interim

    def search_params(self, interations = 20, start_from_all_sensors=False):
        results = pd.DataFrame(columns=['MSE', 'std_MSE', 'alpha', # bigger std means less robust
                                        'epochs', 'nodes', 'dropout',
                                        'activation', 'batch_size',
                                        'sequence_length', 'sensor_length'])

        weights_file = f'{CURRENT_FILE_PATH}/lstm_hyper_parameter_weights.h5'

        alpha_list = [0.01, 0.05] + list(np.arange(10,60+1,10)/100)

        sequence_list = list(np.arange(10,40+1,5))
        epoch_list = list(np.arange(5,20+1,5))
        nodes_list = [[32], [64], [128], [256], [32, 64], [64, 128], [128, 256]]

        # lowest dropout=0.1, because I know zero dropout will yield better training results but worse generalization
        dropouts = list(np.arange(1,5)/10)

        # again, earlier testing revealed relu performed significantly worse, so I removed it from the options
        activation_functions = ['tanh', 'sigmoid']
        batch_size_list = [32, 64, 128, 256]

        sensor_list = self.sensors
        if start_from_all_sensors:
            sensor_list = [[c for c in list(self.df_train) if re.match("sen",c)]]
            sensor_names = [c for c in list(self.df_train) if re.match("sen",c)]
        else:
            sensor_names = self.sensors
            drop_sensors = []
        tuning_options = np.prod([len(alpha_list),
                                  len(sequence_list),
                                  len(epoch_list),
                                  len(nodes_list),
                                  len(dropouts),
                                  len(activation_functions),
                                  len(batch_size_list),
                                  len(sensor_list)])


        mse = []
        best_mse = np.inf
        best_two = {"first": {"mse": np.inf, "model": 0, "history": "", "batch_size":-1, "sequence_length": 10, "sensor_length":11},
            "second": {"mse": np.inf, "model": 0, "history": "", "batch_size":-1, "sequence_length": 10, "sensor_length":11} }

        for i in range(interations):
            if interations < 10:
                print('iteration ', i+1)
            elif ((i+1) % 10 == 0):
                print('iteration ', i+1)

            # init parameters
            alpha = random.sample(alpha_list, 1)[0]
            #sequence_length = random.sample(sequence_list, 1)[0]
            sequence_length = 10
            epochs = random.sample(epoch_list, 1)[0]
            nodes_per_layer = random.sample(nodes_list, 1)[0]
            dropout = random.sample(dropouts, 1)[0]
            activation = random.sample(activation_functions, 1)[0]
            batch_size = random.sample(batch_size_list, 1)[0]

            if start_from_all_sensors:
                remaining_sensors = random.sample(sensor_names, 1)[0]
                remaining_sensors = [remaining_sensors]
                drop_sensors = [element for element in sensor_names if element not in remaining_sensors]
            else:
                remaining_sensors = self.sensors
                drop_sensors = []

            # create model
            input_shape = (sequence_length, len(remaining_sensors))

            # return input_shape, nodes_per_layer, dropout, activation, weights_file

            model = self.build_model_full(input_shape, nodes_per_layer, dropout, activation, weights_file)
            # create train-val split
            X_train_interim, X_test_interim = self.prep_data(self.df_train, self.df_test, drop_sensors, remaining_sensors, alpha)
            gss = GroupShuffleSplit(n_splits=3, train_size=0.80, random_state=42)

            for train_unit, val_unit in gss.split(X_train_interim['id'].unique(), groups=X_train_interim['id'].unique()):
                train_unit = X_train_interim['id'].unique()[train_unit]  # gss returns indexes and index starts at 1
                train_split_array = self.gen_data_wrapper(X_train_interim, sequence_length, remaining_sensors, train_unit)
                #train_split_label = gen_label_wrapper(X_train_interim, sequence_length, ['rul'], train_unit)
                train_split_label = self.gen_label_wrapper(X_train_interim, sequence_length, 'rul', train_unit)

                val_unit = X_train_interim['id'].unique()[val_unit]
                val_split_array = self.gen_data_wrapper(X_train_interim, sequence_length, remaining_sensors, val_unit)
                #val_split_label = gen_label_wrapper(X_train_interim, sequence_length, ['rul'], val_unit)
                val_split_label = self.gen_label_wrapper(X_train_interim, sequence_length, 'rul', val_unit)

                # train and evaluate model
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.load_weights(weights_file)  # reset optimizer and node weights before every training iteration

                history = model.fit(train_split_array, train_split_label,
                                    validation_data=(val_split_array, val_split_label),
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    verbose=0)
                mse_ = history.history['val_loss'][-1]
                mse.append(mse_)
                #mse.append(history.history['val_loss'][-1])
                if mse_ < best_two["second"]["mse"]:
                    if mse_ > best_two["first"]["mse"]:
                        best_two["second"]["mse"] = mse_
                        best_two["second"]["model"] = model
                        best_two["second"]["history"] = history
                        best_two["second"]["sequence_length"] = sequence_length
                        best_two["second"]["sensor_length"] = len(remaining_sensors)
                    else:
                        best_two["second"]["mse"] =  best_two["first"]["mse"]
                        best_two["second"]["model"] = best_two["first"]["model"]
                        best_two["second"]["history"] = best_two["first"]["history"]
                        best_two["second"]["batch_size"] = best_two["first"]["batch_size"]
                        best_two["second"]["sequence_length"] = best_two["first"]["sequence_length"]
                        best_two["second"]["sensor_length"] = best_two["first"]["sensor_length"]

                        best_two["first"]["mse"] = mse_
                        best_two["first"]["model"] = model
                        best_two["first"]["history"] = history
                        best_two["first"]["batch_size"] = batch_size
                        best_two["first"]["sequence_length"] = sequence_length
                        best_two["first"]["sensor_length"] = len(remaining_sensors)

            # append results
            result = {'MSE':np.mean(mse), 'std_MSE':np.std(mse), 'alpha':alpha,
                 'epochs':epochs, 'nodes':str(nodes_per_layer), 'dropout':dropout,
                 'activation':activation, 'batch_size':batch_size, 'sequence_length':sequence_length,
                 'sensor_length':len(remaining_sensors)}
            results = results.append(pd.DataFrame(result, index=[0]), ignore_index=True)

        if not os.path.exists(PRPOJECT_PATH+"/results"):
            os.mkdir(PRPOJECT_PATH+"/results")

        result_file = PRPOJECT_PATH+"/results/nn_lstm_results.csv"
        arch_history = PRPOJECT_PATH+"/results/nn_lstm_history"

        results.to_csv(result_file, index=False)
        for k,v in best_two.items():
            model_file = PRPOJECT_PATH+f"/results/{self.best_model_file_prefix}_{k}.h5"
            v["model"].save(model_file)
            with open(f'{arch_history}_{k}', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            model_param_file = PRPOJECT_PATH+f"/results/{self.best_model_file_prefix}_{k}.csv"
            data_ = pd.DataFrame({"batch_size": [v["batch_size"]], "sequence_length":[v["sequence_length"]],
                    "sensor_length": [v["sensor_length"]]})
            data_.to_csv(model_param_file, index=False)

        return best_two


lst = RulLSTM()
bt = lst.search_params()
results = lst.predict(lst.df_test, model_file=PRPOJECT_PATH+"/results/rul_lstm_n_first.h5", param_file = PRPOJECT_PATH+"/results/rul_lstm_n_first.csv")
print(results)
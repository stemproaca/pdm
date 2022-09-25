import os
import re
import numpy as np
from pathlib import Path
import zipfile
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class DataUtility(object):
    def __init__(self):
        pass

    def unzip_files(self, zip_file_name = None, unzip_to_folder = None, remove_zipped = False):
        if not zip_file_name:
            current_folder = os.getcwd() # or: os.path.dirname(current_folder)
            parent_folder = Path(os.getcwd()).parent.absolute()
            zip_file_name = Path(parent_folder ,  "zipraw/CMAPSS.zip")
        if not unzip_to_folder:
            unzip_to_folder = str(parent_folder) +  "/raw_data"

        if not os.path.exists(unzip_to_folder):
            os.makedirs(unzip_to_folder)

        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(unzip_to_folder)

        if remove_zipped:
            os.remove(zip_file_name)

        return [unzip_to_folder + "/" + file for file in os.listdir(unzip_to_folder)]

    def list_files(self, raw_file_folder = None):
        if not raw_file_folder:
            raw_file_folder = unzip_to_folder = os.getcwd() +  "/raw_data"
        return [raw_file_folder + "/" + file for file in os.listdir(raw_file_folder)]

    def get_files_regex(self, raw_file_folder = None, file_name_str = "test"):
        if not raw_file_folder:
            raw_file_folder = unzip_to_folder = os.getcwd() +  "/raw_data"
        raw_files = self.list_files(raw_file_folder = raw_file_folder)
        regex = re.compile(f".+{file_name_str}.+gz")
        raw_data_files = [f for f in raw_files if re.match(regex, f)]
        return raw_data_files

    def read_data_files(self, raw_file_folder = None, file_name_str = "train", use_pd = True, sep = " ", columns = None):
        if not columns:
            columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
                "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
                ,"sensor20","sensor21" ]

        raw_data_files = self.get_files_regex(raw_file_folder = raw_file_folder, file_name_str =file_name_str)

        df_total =  pd.DataFrame()
        for f in raw_data_files:
            if use_pd:
                df_ = pd.read_csv(f, compression='gzip',index_col = False, names = columns, sep=' ')
            else:
                df_= pd.DataFrame(np.loadtxt(f), columns=columns)
            df_[["id", "cycle"]] = df_[["id", "cycle"]].astype(int)

            flag = re.findall(r"FD\d{3}", str(f))[0]
            df_["Flag"] = flag
            if df_total.empty:
                df_total = df_.copy()
            else:
                df_total = pd.concat([df_total, df_], axis = 0 )

        return df_total

    def read_result(self, raw_file_folder = None, file_name_str = "RUL_FD", use_pd = True, sep = " ", columns = None):
        raw_data_files = self.get_files_regex(raw_file_folder = raw_file_folder, file_name_str =file_name_str)
        if not columns:
            columns = ["rul"]

        df_result =  pd.DataFrame()
        for f in raw_data_files:
            if use_pd:
                df_ = pd.read_csv(f, compression='gzip', index_col = False, names = columns, sep = sep)
            else:
                df_= pd.DataFrame(np.loadtxt(f), columns = columns)
            flag = re.findall(r"FD\d{3}", str(f))[0]
            df_["Flag"] = flag
            if df_result.empty:
                df_result = df_.copy()
            else:
                df_result = pd.concat([df_result, df_], axis = 0 )
        return df_result

    def prepare_dfs(self, raw_file_folder = None, use_pd = True, sep = " "):
        if not raw_file_folder:
            raw_file_folder = unzip_to_folder = os.getcwd() +  "/raw_data"

        raw_files = self.list_files(raw_file_folder = raw_file_folder)

        columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
            "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
            ,"sensor20","sensor21" ]

        # Train
        df_train = self.read_data_files(raw_file_folder = raw_file_folder,
            file_name_str = "train", use_pd = use_pd, sep = " ", columns = columns)
        # Test
        df_test = self.read_data_files(raw_file_folder = raw_file_folder,
            file_name_str = "test", use_pd = use_pd, sep = " ", columns = columns)

        resul_columns = ["rul"]
        df_result = self.read_result(raw_file_folder = raw_file_folder, file_name_str = "RUL_FD", \
            use_pd = use_pd, sep =sep, columns = resul_columns)

        df_train.iloc[:, [0,1]] = df_train.iloc[:, [0,1]].astype(int)
        df_test.iloc[:, [0,1]] = df_test.iloc[:, [0,1]].astype(int)

        df_max = df_test.groupby(["Flag","id"])["cycle"].max().reset_index()
        df_result = df_result.reset_index()
        df_result["id"] = df_result.groupby("Flag")["index"].rank("first", ascending = True).astype(int)
        df_result.drop(columns = ["index"], inplace = True)

        df_result = df_result.merge(df_max, on = ["Flag", "id"], how = "inner")

        df_result["rul_failed"] = df_result["rul"] + df_result["cycle"]

        df_test = df_test.merge(df_result[["rul_failed", "Flag", "id"]], on = ["Flag", "id"], how = "inner")
        df_test["remaining_rul"] = df_test["rul_failed"] - df_test["cycle"]

        #df_test[["rul_failed", "remaining_rul"]] = df_test[["rul_failed", "remaining_rul"]].astype(int)
        return df_train, df_test, df_result
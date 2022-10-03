import os
import re
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.dirname(CURRENT_FILE_PATH)

"""
    This is data loading module
"""
class DataUtility(object):
    """
        Data Utility:
        constructor: specify raw data folder.
        it has the unzip function as well
    """
    def __init__(self, parent_data_folder_ = None):
        if parent_data_folder_:
            self.parent_data_folder = parent_data_folder_
        else:
            self.data_parent_folder = FILE_PATH

        self.FILE_PATHS = self.get_file_paths_()

    def get_file_paths_(self):
        file_paths = {}
        file_paths["parent_folder"] = self.data_parent_folder
        file_paths["raw_data_path"] = self.data_parent_folder + '/raw_data'
        file_paths["zip_data_path"] = self.data_parent_folder + '/zipraw'
        file_paths["unzip_to_path"] = self.data_parent_folder + '/raw_data'
        return file_paths


    def unzip_files(self, zip_file_name = None,  remove_zipped = False):
        if not zip_file_name:
            zip_file_name = f'{self.FILE_PATHS["zip_data_path"]}/CMAPSS.zip'

        if not os.path.exists(self.FILE_PATHS["zip_data_path"]):
            os.makedirs(self.FILE_PATHS["zip_data_path"])

        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(self.FILE_PATHS["unzip_to_path"])

        if remove_zipped:
            os.remove(zip_file_name)

        return [self.FILE_PATHS["unzip_to_path"] + "/" + file for file in os.listdir(self.FILE_PATHS["unzip_to_path"])]

    def list_data_files(self):
        return [self.FILE_PATHS["raw_data_path"] + "/" + file for file in os.listdir(self.FILE_PATHS["raw_data_path"])]

    def get_files_regex(self, file_name_str = "test"):
        raw_files = self.list_data_files()
        regex = re.compile(f".+{file_name_str}.+gz")
        raw_data_files = [f for f in raw_files if re.match(regex, f)]
        return raw_data_files


    def read_data_files(self, file_name_str = "train", use_pd = True, sep = " ", columns = None):
        if not columns:
            columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
                "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
                ,"sensor20","sensor21" ]

        raw_data_files = self.get_files_regex(file_name_str =file_name_str)

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


    def read_result(self, file_name_str = "RUL_FD", use_pd = True, sep = " ", columns = None):

        raw_data_files = self.get_files_regex(file_name_str =file_name_str)
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


    def prepare_dfs(self, use_pd = True, sep = " "):
        """
            this function generates all the data for our project
            input: use_pd -- use pandas or not
                    sep -- field separator
            output: df_train, df_test, df_result
        """
        columns=["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
            "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
            ,"sensor20","sensor21" ]

        # Train
        df_train = self.read_data_files( file_name_str = "train", use_pd = use_pd, sep = " ", columns = columns)
        # Test
        df_test = self.read_data_files( file_name_str = "test", use_pd = use_pd, sep = " ", columns = columns)

        resul_columns = ["rul"]
        df_result = self.read_result(file_name_str = "RUL_FD", \
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

""" test
if __name__ == "__main__":
    data_loader = DataUtility()
    df_train, df_test, df_result = data_loader.prepare_dfs()
"""
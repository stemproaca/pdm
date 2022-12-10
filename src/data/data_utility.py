import os
import re
import zipfile
import glob
import shutil
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CURRENT_FILE_PATH_PARENT = os.path.dirname(CURRENT_FILE_PATH)
DATAFOLDER = os.path.dirname(CURRENT_FILE_PATH_PARENT)

# Data Utility
class DataUtility(object):
    """
    This utility handles unzipping raw data, read raw data into pandas dataframes.
    Args:
        object (_type_): _description_
    """
    def __init__(self,
    parent_data_folder=None,
    raw_data_folder=None,
    zipped_folder=None,
    unzip_to=None,
    train_columns=None,
    test_columns=None,
    result_columns=None,
    **kwargs):
        """
        specify where data files are stored and where unzip and move to
        Args:
            parent_data_folder (_type_, optional): parent folder of this file. in this project it is /data. Defaults to None.
            raw_data_folder (_type_, optional): raw data folder. Defaults to None.
            zipped_folder (_type_, optional): zipped files folder. Defaults to None.
            unzip_to (_type_, optional): unzip to folder. user can choose to move unzipped files to raw data folder. Defaults to None.
        """
        if not parent_data_folder:
            parent_data_folder = DATAFOLDER + "/data"
        self.parent_data_folder = parent_data_folder
        self.file_folders = {}
        self.file_folders["raw_data_folder"] = self.parent_data_folder+'/raw' if not raw_data_folder else raw_data_folder
        self.file_folders["zipped_data_folder"] = self.parent_data_folder+'/zipped' if not zipped_folder else zipped_folder
        self.file_folders["unzip_to_folder"] = self.parent_data_folder+'/raw' if not raw_data_folder else unzip_to
        if not train_columns:
            train_columns = ["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
                "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
                ,"sensor20","sensor21" ]
        self.train_columns = train_columns

        if not test_columns:
            test_columns = ["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8",
                "sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
                ,"sensor20","sensor21" ]
        self.test_columns = test_columns

        if not result_columns:
            result_columns = ["rul"]
        self.result_columns = result_columns

        # prepare unzip_to_folder folders
        if not os.path.exists(self.file_folders["unzip_to_folder"]):
            os.mkdir(self.file_folders["unzip_to_folder"])

    # Unzipping
    def unzip_file(self, zip_file_name, remove_zipped = False, move_to_raw_folder = True, overwrite=True):
        """_summary_

        Args:
            zip_file_name (_type_): full file name.
            remove_zipped (bool, optional): whether to delete zip files after unzip. Defaults to False.
            move_to_raw_folder (bool, optional): whether move unzipped files to raw data folder. Defaults to True.
            overwrite (bool, optional): specify whether overwrite raw data if a same name aready exists. Defaults to True.
        """
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(self.file_folders["unzip_to_folder"])
        if remove_zipped:
            os.remove(zip_file_name)

        if move_to_raw_folder & (self.file_folders["raw_data_folder"]!=self.file_folders["unzip_to_folder"]):
            unzipped_files = glob.glob(os.path.join(self.file_folders["unzip_to_folder"], '*_A_*'), recursive=True)

            # iterate on all files to move them to destination folder
            for file_path in unzipped_files:
                destination_path = os.path.join(self.file_folders["raw_data_folder"], os.path.basename(file_path))
                # will over write
                if (not os.path.exists(destination_path))  or overwrite:
                    shutil.move(file_path, destination_path)

    # Unzipping
    def uzip_files(self, zip_files = None, remove_zipped=False):
        """_summary_

        Args:
            zip_files (_type_, optional): user can choose/supply other zip files. Defaults to None.
            remove_zipped (bool, optional): whether to delete zip files after unzip. Defaults to False.
        """
        if not zip_files:
            zip_files = os.listdir(self.file_folders["zipped_data_folder"])
            zip_files = [self.file_folders["zipped_data_folder"] + f for f in zip_files if re.match(r".zip",f) ]
        else:
            if isinstance(zip_files,str):
                zip_files = [zip_files]

        for one_zip_file in zip_files:
            self.unzip_file(one_zip_file, remove_zipped=remove_zipped)

    # Listing
    def list_raw_data_files(self):
        """
        List raw data files

        Returns:
            _type_: list available raw data files
        """
        return [self.file_folders["raw_data_folder"] + "/" + file for file in os.listdir(self.file_folders["raw_data_folder"])]

    # Listing
    def get_files_regex(self, file_name_str = "test"):
        """
        filter files list and return list of files with file names matching certain pattern

        Args:
            file_name_str (str, optional): file name pattern to match. Defaults to "test".

        Returns:
            _type_: list of files with file names matching certain pattern
        """
        raw_files = self.list_raw_data_files()
        regex = re.compile(f".+{file_name_str}.+gz")
        raw_data_files = [f for f in raw_files if re.match(regex, f)]
        return raw_data_files


    # Reading
    def read_data_files(self, file_name_str = "train", use_pd = True, sep = " "):
        """
        Read raw data file by type of files, combine them and return a single data frame

        Args:
            file_name_str (str, optional): file name pattern. Defaults to "train". another: test
            use_pd (bool, optional): method to generate dataframe. pd - user pandas, otherwise, use nump utility . Defaults to True.
            sep (str, optional): column/field seperator. Defaults to " ".
            columns (_type_, optional): names for data column. Defaults to None.

        Returns:
            _type_: Pandas dataframe
        """
        columns = self.train_columns if file_name_str=="train" else self.test_columns
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

    # Reading
    def read_result(self, file_name_str = "RUL_FD", use_pd = True, sep = " "):
        """
        utility to read result file.
        Args:
            file_name_str (str, optional): file pattern. Defaults to "RUL_FD".
            use_pd (bool, optional): method to generate dataframe. pd - user pandas, otherwise, use nump utility . Defaults to True.
            sep (str, optional): column/field seperator. Defaults to " ".
            columns (_type_, optional): names for data column. Defaults to None.

        Returns:
            _type_: Pandas dataframe
        """

        raw_data_files = self.get_files_regex(file_name_str =file_name_str)

        df_result =  pd.DataFrame()
        for f in raw_data_files:
            if use_pd:
                df_ = pd.read_csv(f, compression='gzip', index_col = False, names = self.result_columns, sep = sep)
            else:
                df_= pd.DataFrame(np.loadtxt(f), columns = self.result_columns)
            flag = re.findall(r"FD\d{3}", str(f))[0]
            df_["Flag"] = flag
            if df_result.empty:
                df_result = df_.copy()
            else:
                df_result = pd.concat([df_result, df_], axis = 0 )
        return df_result

    # Reading
    def prepare_dfs(self, use_pd = True, sep = " ", train_pattern="train", test_pattern="test", result_pattern="RUL_FD" ):
        """
        this function generates all the data for our project

        Args:
            use_pd (bool, optional): method to generate dataframe. pd - user pandas, otherwise, use nump utility . Defaults to True.
            sep (str, optional): column/field seperator. Defaults to " ".

        Returns:
            _type_: train, test and result dataframes
        """
        # Train
        df_train = self.read_data_files( file_name_str = train_pattern, use_pd = use_pd, sep = " ")
        # Test
        df_test = self.read_data_files( file_name_str =test_pattern, use_pd = use_pd, sep = " ")

        df_result = self.read_result(file_name_str = result_pattern, \
            use_pd = use_pd, sep =sep)

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

    # Reading
    def load_data_by_flags(self, flags = None):
        """
        wrapping prepare_dfs method and return data frames for specific flags

        Args:
            flags (_type_, optional): flags is type of data, In this project, we have FD001-004. Defaults to None.

        Returns:
            _type_: train, test and result dataframes
        """
        df_train, df_test, df_result = self.prepare_dfs()
        df_train["rul"] = df_train.groupby(["Flag","id"])["cycle"].transform("max")-df_train["cycle"]

        if not flags:
            return df_train, df_test, df_result
        if type(flags) == str:
            flags = [flags]
        return df_train[df_train["Flag"].isin(flags)],df_test[df_test["Flag"].isin(flags)],df_result[df_result["Flag"].isin(flags)]

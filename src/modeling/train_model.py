import sys
import os


SCRIPT_DIR = os.environ["SCRIPT_DIR"]

sys.path.append(SCRIPT_DIR)
from preparing.data_utility import DataUtility
from featurizing.feature_selection_eda import SelectingEDAFeatures
from serving.excel_reporting import ExcelUtility
from modeling.rul_lstm import RulLSTM

def train(use_model):
    if use_model == "lstm":
        print("start training lstm model")
        model = RulLSTM()
        model.full_train()
        print(f'Models {os.environ["MODEL_FILE_FIRST"]} and {os.environ["MODEL_FILE_SECOND"]} saved to {os.environ["MODEL_DIR"]}')
    else:


from  sklearn.feature_selection import VarianceThreshold
from collections import Counter
from copy import deepcopy 
from dtaidistance import dtw, dtw_ndim
from fitter import Fitter, get_common_distributions, get_distributions
from IPython.display import display, Markdown, Image
from kneed import DataGenerator, KneeLocator
from scipy.fft import fft, ifft
from scipy.stats import pearsonr 
from sklearn import decomposition
from sklearn import ensemble 
from sklearn import kernel_ridge
from sklearn import linear_model 
from sklearn import model_selection
from sklearn import svm 
from sklearn import tree  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import scale
from sklearn.utils import class_weight
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller 
import catboost  
import lightgbm 
import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as mn 
import numpy as np
import os
import pandas as pd
import random
import re
import seaborn as sns
import sys
import time   
import warnings

class SelectingEDAFeatures(object):
    def __init__(self, per_missing=0.50, dominant_threshold= 0.95, \
                dispersion_threshold=(1-1.0e-9, 1+1.0e-9), heat_map_threshold = 0.9,\
                max_vif=9): 
        """
            documentation: 
            set thresholds
        """ 
        self.per_missing = per_missing  
        self.dominant_threshold = dominant_threshold
        self.dispersion_threshold = dispersion_threshold
        self.all_feature_columsn = [col for col in list(DF_TRAIN) if re.match("(op|sen)", col)]
        self.heat_map_threshold = heat_map_threshold
        self.max_vif = max_vif
    
    def eda_features(self):
        """
            doc:
            output: conduct exploration and summarize
        """
    def indentify_missing(self):
        """
            documentation: 
            identify missing
            missingno
        """ 
        tmp = DF_TRAIN.copy()  
        #masks = np.random.choice([True, False], size=tmp.shape, p = [p, 1-p])
        #tmp = tmp.mask(masks)    
        return np.array(list(tmp))[tmp.isnull().sum(axis=0)/tmp.shape[0] > self.per_missing]

    def dominant(self, x, per): 
        """
            documentation: 
            helper fun
        """ 
        return Counter(x).most_common(1)[0][1]/len(x) > per

    def identify_constants(self):
        """
            documentation: 
            identify constants
        """ 
        tmp = DF_TRAIN.copy()
        tmp = tmp.select_dtypes(include = "number")
        tmp.drop(columns = ["id", "cycle", "rul"], inplace = True)

        feats =  list(tmp)

        # compare it in two way: threshold = 0  
        vt = VarianceThreshold(threshold=0) #恒值
        _ = vt.fit_transform(tmp)
        kept_columns = vt.get_feature_names_out() 

        consts = [c for c in feats if c not in kept_columns] 

        # dominant values doesn't account for per%   
        quasi_columns = np.array(list(tmp))[tmp.apply(lambda x: self.dominant(x, self.dominant_threshold))]

        return list(set(consts + list(quasi_columns))) 
        #ROUNDONE_CONST = round_one_remove_low_variance()  
                 
    def plot_columns(self):
        """
            documentation: 
            plotting
        """ 
        tmp = DF_TRAIN.copy()
        tmp = tmp.select_dtypes(include = "number")
        tmp.drop(columns = ["id", "cycle", "rul"], inplace = True) 
        tmp.hist(bins=30, figsize=(15, 10),   layout=(-1, 4) ) 
        plt.tight_layout()
        plt.show()

    def dispersion(self,data): 
        """
            documentation: 
            dispersions. 
            return flat cols and dispersion values 
        """ 
        arith_mean = np.mean(data+1, axis =0 )
        #geo_mean = np.power(np.prod(data, axis =0 ),1/data.shape[0]) 
        geo_mean = sum(np.log1p(data))/len(data)
        geo_mean = np.exp(geo_mean) 
        return arith_mean/geo_mean  

    def identify_flat_dispersion(self):
        tmp = DF_TRAIN.copy()
        tmp = tmp.select_dtypes(include = "number")
        tmp.drop(columns = ["id", "cycle", "rul"], inplace = True)

        dispersions = tmp.apply(lambda x: self.dispersion(x), axis=0)
        criteria = np.where(np.logical_and(dispersions>self.dispersion_threshold[0], \
                                           dispersions<self.dispersion_threshold[1]))
        low_dispersion = np.array(list(tmp))[criteria]
        return low_dispersion, dispersions
        #dispersions = np.array(list(tmp))[tmp.apply(lambda x: dispersion(x), axis=0)>threshold]  
        #ROUNDONE_DISPERSION, _  = filter_by_dispersion(threshold=thresholds)
        
    def plot_by_id_corner(self, ids = [12,76], n_cols =5, pre_filtered_cols = None):
        """
            documentation: 
            plotting corr
        """ 
        if not pre_filtered_cols:
            pre_filtered_cols = self.cols_for_plotting()

        tmp = DF_TRAIN.copy()  
        tmp = tmp[tmp["id"].isin(ids)]

        max_rul = tmp["rul"].max()
        for id_engine in ids:
            if max_rul > tmp[tmp["id"]==id_engine]["rul"].max():
                max_rul = tmp[tmp["id"]==id_engine]["rul"].max()
        tmp = tmp[tmp["rul"]<=max_rul]  
  
        cols = random.sample(pre_filtered_cols, n_cols)
 
        cols.append("rul") 

        # seaborn matplotlib
        g = sns.PairGrid(tmp, vars = cols,diag_sharey = False, corner = True, hue = 'id' ) 
        g.map_lower(plt.scatter, alpha=0.6)
        #g.map_diag(plt.hist, alpha=0.7 ) 
        g.map_diag(sns.kdeplot)

        plt.show()   
        #plot_by_id_corner()
    
    def plot_by_id_pair(self, ids = [2,76], n_cols =5, pre_filtered_cols = None):
        """
            documentation: 
            plotting corr
        """         
        sns.set_style("whitegrid") 
        
        if not pre_filtered_cols:
            pre_filtered_cols = self.cols_for_plotting()

        tmp = DF_TRAIN.copy()  
        tmp = tmp[tmp["id"].isin(ids)]

        max_rul = tmp["rul"].max()
        for id_engine in ids:
            if max_rul > tmp[tmp["id"]==id_engine]["rul"].max():
                max_rul = tmp[tmp["id"]==id_engine]["rul"].max()
        tmp = tmp[tmp["rul"]<=max_rul]  

        cols = random.sample(pre_filtered_cols, n_cols) 
        cols.append("rul") 

        sns.pairplot(tmp, vars=cols, diag_kind = 'kde', hue="id")

        plt.show()   
        #plot_by_id_pair()

    def plot_by_id_strip(self, ruls=None, n_feat=5, pre_filtered_cols=None):
        """
            documentation: 
            plotting corr
        """         
        sns.set_style("whitegrid") 
        
        if not pre_filtered_cols:
            pre_filtered_cols = self.cols_for_plotting() 

        tmp = DF_TRAIN.copy() 

        if not ruls:
            ruls = [i for i in range(1, tmp["rul"].max(), 50)]

        tmp = tmp[tmp["rul"].isin(ruls)]

        cols = random.sample(pre_filtered_cols, n_feat)  

        sns.stripplot(data=tmp, y="sensor21", x="rul",  dodge=True, jitter=False,  palette="deep")

        plt.show()   
        return tmp    
    
    def cols_for_plotting(self):
        """
            doc:
            helper function
        """
        constant_cols = self.identify_constants() 
        return [c for c in self.all_feature_columsn if c not in constant_cols] 
    
    def heatmap_plot(self, show_fig=True, pre_filtered_cols=None ):
        """
            doc:
            return high corr cols
        """  
        if not pre_filtered_cols:
            pre_filtered_cols = self.cols_for_plotting()
            
        tmp = DF_TRAIN.copy()  
        tmp_corr = tmp[pre_filtered_cols].corr(method="pearson")
        
        if show_fig:
            plt.figure(figsize=(9, 9))

            ax = sns.heatmap(tmp_corr,  square=True, annot=True,annot_kws={"size": 8},
                        center=0, fmt=".2f",   linewidths=.5,  #fmt=".2g",
                        cmap="vlag", cbar_kws={"shrink": 0.8});

            ax.xaxis.tick_top()
            #sns.set(font_scale=1)
            plt.xticks(rotation = 90)
            plt.show()
  
        colss = list(tmp_corr)
        ln = len(tmp_corr)  
        allvals = tmp_corr.values
        pairs = []
        for i in range(ln):
            for j in range(i+1, ln):
                if abs(allvals[i, j])>self.heat_map_threshold:
                    pairs.append((colss[i], colss[j], allvals[i, j]))
        return pairs
 
    def filter_by_val(self, feats):   # 9.0 is conventional chosen 
        """
            doc:
            return high corr cols
        """   
        ss = StandardScaler() # Normalizer, RobustScaler, minmax_scale
        tmp = DF_TRAIN.copy()   
        tmp = tmp[feats]  

        ss_tmp = ss.fit_transform(tmp) 

        ss_tmp = pd.DataFrame(ss_tmp, columns=list(tmp)) 
        vif_data = pd.DataFrame()
        vif_data["feature"] = ss_tmp.columns

        vif_data["VIF"] = [variance_inflation_factor(ss_tmp.values, i) for i in range(len(ss_tmp.columns))]

        return list(vif_data["feature"]), vif_data  
  
    def identify_high_vif_columns(self, pre_filtered_cols=None):
        """
            doc:
            return high corr cols from VIF
            return 
        """  
        if not pre_filtered_cols:
            pre_filtered_cols = self.cols_for_plotting()
            
        tmp = DF_TRAIN.copy()   
        val = np.inf
        rounds = 0 
        max_rounds = tmp.shape[1]
        
        feats = deepcopy(pre_filtered_cols) 
        high_vif_col = [] 
    
        while val > self.max_vif:  
            rounds += 1  
            feats, df_vals = self.filter_by_val(feats=feats) 
      
            max_row = df_vals.loc[df_vals["VIF"].idxmax()]

            feat, val = max_row[0], max_row[1]

            if val < self.max_vif:
                return list(feats), high_vif_col 
            else: 
                feats.remove(feat)
                high_vif_col.append(feat)
            if rounds>max_rounds:
                break
        return feats, high_vif_col
        
    def eda_features(self):
        """
            doc:
            output: conduct exploration and summarize
        """
        missing = self.indentify_missing()
        const = self.identify_constants()
        flats, _ = self.identify_flat_dispersion()
        _, vif = self.identify_high_vif_columns()
        
        to_remove = list(missing) + list(const) + list(flats) + list(vif) 
      
        return [c for c in self.all_feature_columsn if c not in to_remove]
    
        

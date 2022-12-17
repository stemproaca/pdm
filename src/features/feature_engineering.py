import os
import re
import sys
import openpyxl
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
from numpy import isnan
import missingno as mn
from scipy import linalg
from statistics import mode
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.manifold import TSNE
from openpyxl import load_workbook
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
from scipy.spatial.distance import pdist
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import squareform
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from IPython.display import display, Markdown, Image
from sklearn.preprocessing import QuantileTransformer
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler


CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CURRENT_FILE_PATH_PARENT = os.path.dirname(CURRENT_FILE_PATH)
PRPOJECT_PATH = os.path.dirname(CURRENT_FILE_PATH_PARENT)

sys.path.append(CURRENT_FILE_PATH_PARENT)
from data.data_utility import DataUtility
from features.feature_selection_eda import SelectingEDAFeatures
from serving.excel_reporting import ExcelUtility

class FeatureResearch(ExcelUtility, DataUtility):
    """
    Common Techniques
        1. Imputation
        2. Handling Outliers
        3. Binning
        4. Transformation and Scaling
        5. Categorical: Ordinal, Nominal
        6. Grouping Operations
        7. Feature Split
        9. Extracting Date
    Advanced Techniques：
        1. IterativeImputer
        2. IsolationForest
        3. MissForest from missingpy
        4. Time series feature engineering: lag, rolling, expanding etc.
    """
    def __init__(self,  flags=None, excel_file_prefix=None):
        if not flags:
            flags = "FD001"
        if not excel_file_prefix:
            excel_file_prefix = "FeatureResearch"
        ExcelUtility.__init__(self, excel_file_prefix=excel_file_prefix)
        DataUtility.__init__(self)
        self.train, self.test, self.result = self.load_data_by_flags(flags=flags)
        self.prepare_excel_report()

    def utility(self):
        # masking
        vals = np.random.choice([True,False], size=self.train.shape, p=[0.2, 0.8])
        df_vals = pd.DataFrame(vals, columns=list(self.train))
        idf1 = self.train.mask(df_vals)

        #filling
        idf1 = idf.mask(np.random.choice([False, True], size=idf.shape, p=[0.2, 0.7]))
        idf2 = idf1.fillna(method='bfill')

        # idxmax()
        idf1["sensor11"].value_counts().idxmin()

        # curve_fit
        x = idf["sensor21"]
        y = idf["rul"]
        f3 = interp1d(x,y,kind='cubic')
        scipy.optimize.curve_fit

        # x values can not be the same
        f1 = interp1d(x, y, kind='linear')
        f3 = interp1d(x, y, kind='cubic')

    def get_df(self):
        df = self.train.copy()
        return df.mask(np.random.choice([False, True], size=df.shape, p=[0.82,.18]))

    def fake_prodce_missing(self,  df=pd.DataFrame(),  upper_limit=0.25, \
                            show_png=False, to_plot=True, drop_limit=0.20, to_drop=False):
        if df.empty:
            tmp = self.train.copy()
        else:
            tmp = self.train.copy()

        sn = "Missing"
        startn = 2
        # few lines
        self.excel_add_data(tmp.sample(10), sn=sn, start_col=1, start_row=startn, title="Data Scenario - Missing")

        startn += 1
        startn += 10

        plt.ioff()
        # upper_limit: percent of "masked" -- > None
        # prepare a fake data set
        if upper_limit > 1:
            upper_limit = upper_limit/100

        len_col = tmp.shape[0]

        for c in list(tmp)[3:]:
            p = np.random.uniform(low = 0.0,high = upper_limit, size = None) # size int or tuple
            num = np.zeros(len_col)
            #tfs = np.random.choice([True, False], size=num.shape, p=[p,1-p])
            tfs = np.random.choice([True, False], size=len_col, p=[p,1-p])
            tmp[c] = tmp[c].mask(tfs)

        graph_height = 5
        rows_graph = 0
        startn += 2
        if to_plot:
            rows_graph = int(graph_height * 5)

            tmp.isna().mean().sort_values().plot(
                kind="bar", figsize=(15,graph_height),
                title = "fake data - percent of missing values per feature",
                ylabel = "ratio of missing values per feature")

            plt.tight_layout()
            plt.savefig(self.tmp_png)
            self.excel_chart(sn=sn, start_row=startn, title="Missing Feature Plotting")
            startn += 1
            # other methods:
            # four types of missing chart : mn.matrix(test_df, figsize=(8,3)), mn.heatmap(test_df, figsize=(10,8))
            # mn.dendrogram(test_df, figsize=(8,4)) or
            # simply plt.imshow(test_df.isna(), aspect="auto", interpolation = "nearest", cmap = "Accent")
            # mn.bar(df_demo, figsize=(8,5), fontsize=10)
            # mn.matrix(df_demo, labels=True, sort="descending", figsize=(12,8));
            # mn.matrix(test_df, figsize=(8,3))

            # missingno matrix
            startn += int(graph_height * 5)
            mn.matrix(tmp, labels=True, sort="descending", figsize=(12,graph_height))
            plt.tight_layout()
            plt.savefig(self.tmp_png)
            self.excel_chart(sn=sn, start_row=startn, title="Missing -- Matrix Display")
            startn += 1

            # missingno dendrogram
            startn += int(graph_height * 5)
            mn.dendrogram(tmp,  figsize=(12,graph_height))
            plt.tight_layout()
            plt.savefig(self.tmp_png)
            self.excel_chart(sn=sn, start_row=startn, title="Missing -- Dendrogram")
            startn += 1

            # missingno bar
            startn += int(graph_height * 5)
            mn.bar(tmp,  figsize=(12,graph_height), fontsize=10)
            plt.tight_layout()
            plt.savefig(self.tmp_png)
            self.excel_chart(sn=sn, start_row=startn, title="Missing -- Bar")
            startn += 1

            # missingno heatmap
            startn += int(graph_height * 5)
            mn.heatmap(tmp,  figsize=(12,graph_height))
            plt.tight_layout()
            plt.savefig(self.tmp_png)
            self.excel_chart(sn=sn, start_row=startn, title="Missing -- Heatmap")
            startn += 1


            # imshow
            startn += int(graph_height * 5)
            plt.imshow(tmp.isna(), aspect="auto", interpolation = "nearest", cmap = "Accent")
            plt.tight_layout()
            plt.savefig(self.tmp_png)
            self.excel_chart(sn=sn, start_row=startn, title="Matplotlib imshow")

            startn += int(graph_height * 5)
            startn += 1

            if show_png:
                plt.show()
            plt.close()

        # columns' per missing
        df_missing = pd.DataFrame( {"Missing Count": tmp.isnull().sum()})
        df_missing["Rows"] = df_missing.shape[0]
        df_missing["Missing %"] = df_missing["Missing Count"] * 100./ df_missing["Rows"]
        self.excel_add_data(df_missing, sn=sn, start_col=1, start_row=startn, title="Data Scenario - Missing")
        startn += df_missing.shape[0]
        startn += 3

        # columns
        list_cols = list(tmp)
        self.excel_add_data(list_cols, sn=sn, start_col=1, start_row=startn, title="All Columns")
        startn += 2
        high_cols = list(tmp.columns[tmp.isnull().mean()>drop_limit])
        self.excel_add_data(high_cols, sn=sn, start_col=1, start_row=startn, title=f"Columns To Be Dropped -- {drop_limit} ")

        if to_drop:
            tmp.dropna(thresh=drop_limit,how='all',axis=1)
            #tmp.dropna(thresh=df.shape[0]*0.6,how='all',axis=1)

        return [c for c in list_cols if c not in high_cols], tmp

    def fake_imputation(self, df=pd.DataFrame(), show_png=False, plot_col=None):
        # 1. SimpleImputer(missing_values=nan,strategy="mean"),
        # 2. KNNImputer(n_neighbors=2, weights="uniform")
        # 3. for SimpleImputer, the impu.statistics_ lists the statisitc for imputer
        # 4. fill mode tmp[col] = tmp[col].fillna(tmp[col].mode()[0])
        # 5. other methods : {'backfill', 'bfill', 'pad', 'ffill', None}
        # 6. from sklearn_pandas import CategoricalImputer. same as most frequent
        # from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
        _, tmp = self.fake_prodce_missing(df=df, show_png=False, to_plot=False, to_drop=False)

        sn, startn = "Imputation", 2
        graph_height = 5
        rows_graph = int(graph_height * 5)

        # SimpleImputer
        df1 = tmp.copy()

        simi_num = SimpleImputer(strategy="mean")
        simi_obj = SimpleImputer(missing_values=None, strategy="most_frequent")

        imputers = {}

        idf = simi_num.fit_transform(df1.select_dtypes(include="number"))
        idf = pd.DataFrame(idf, columns=list(df1.select_dtypes(include="number")))

        tmp1 = df1.select_dtypes(include="object")
        tmp1 = tmp1.where(pd.notnull(tmp1), None)
        simi_obj = SimpleImputer(missing_values=None, strategy="most_frequent")
        _ = simi_obj.fit_transform(tmp1)

        imputers["number_columns"] ={}
        for i, col in enumerate(list(df1.select_dtypes(include="number"))):
            imputers["number_columns"][col] = simi_num.statistics_[i]

        imputers["string_columns"] ={}
        for i, col in enumerate(list(tmp1.select_dtypes(include="object"))):
            imputers["string_columns"][col] = simi_obj.statistics_[i]

        dff = pd.DataFrame.from_dict(imputers,orient='columns').reset_index()
        self.excel_add_data(dff, sn=sn, start_col=1, start_row=startn, title=f"Imputation Statistics")

        startn += dff.shape[0] + 1
        #compare before/after
        if not plot_col:
            plot_col = "sensor4"

        tmp_plot = pd.DataFrame({"Original": df1[plot_col], "mean_imputed": idf[plot_col]})
        tmp_plot.plot(kind="kde", figsize=(12,graph_height), title=f"Imputation {plot_col}")
        plt.savefig(self.tmp_png)
        self.excel_chart(sn=sn, start_row=startn, title=f"Imputation {plot_col}")

        if show_png:
            plt.show()
        plt.close()

        ### use KNN Imputer.
        df1 = tmp.copy()
        knn = KNNImputer(n_neighbors=5, weights="uniform") #use eucliden distance
        df1 = df1.select_dtypes(include="number")

        return imputers, idf

    def fake_advanced_impute(self):
        """
            1. class sklearn.impute.IterativeImputer: Multivariate imputer that estimates each feature from all the others.
                sklearn IterativeImputer is imported from fancyimputer
                class sklearn.impute.MissingIndicator
            2. fancyimpute
            3. very time consuming
        """
        _, tmp = self.fake_prodce_missing(df=df, show_png=False, to_plot=False, to_drop=False)
        knnImpute = KNN(k=3)
        df1 = tmp.copy()

        df1 = tmp.select_dtypes(include="number")

        X_filled_knn = knnImpute.fit_transform(df1)
        X_filled_nnm = NuclearNormMinimization().fit_transform(df1)

        # Instead of solving the nuclear norm objective directly, instead
        # induce sparsity using singular value thresholding
        softImpute = SoftImpute()

        # simultaneously normalizes the rows and columns of your observed data,
        # sometimes useful for low-rank imputation methods
        biscaler = BiScaler()

        # rescale both rows and columns to have zero mean and unit variance
        X_incomplete_normalized = biscaler.fit_transform(df1)

        X_filled_softimpute_normalized = softImpute.fit_transform(X_incomplete_normalized)
        X_filled_softimpute = biscaler.inverse_transform(X_filled_softimpute_normalized)

        X_filled_softimpute_no_biscale = softImpute.fit_transform(df1)

        meanfill_mse = ((X_filled_mean[missing_mask] - X[missing_mask]) ** 2).mean()
        print("meanFill MSE: %f" % meanfill_mse)

        # print mean squared error for the imputation methods above
        nnm_mse = ((X_filled_nnm[missing_mask] - X[missing_mask]) ** 2).mean()

    def fake_interpolation(self, iid=51, feat="sensor4", show_png=False, flag="FD001", flag_col = "Flag"):
        """
            polynomial methods
            Lagrange Polynomial Interpolation
            Newton Polynomial Interpolation, also called Newton’s divided differences interpolation polynomial
            Spline Interpolation and more specifically Cubic Spline Interpolation
        """
        sn = "Interpolation"
        startn = 2
        graph_height = 5
        rows_graph = int(graph_height * 5)

        tmp = self.train.copy()
        tmp = tmp.mask(np.random.choice([True, False], size=tmp.shape, p=[0.2, 0.8]))

        tmp = tmp[(tmp["id"]==iid)&(tmp[flag_col]==flag)]
        tmp_data = tmp[feat]
        interpolated = tmp_data.interpolate(method="polynomial", order=2)
        original = self.train[(self.train["id"]==iid)&(self.train[flag_col]==flag)][feat]

        df_inter = pd.concat([tmp_data, interpolated], axis=1)
        df_inter.columns=["before", "after"]
        df_inter = df_inter.reset_index(drop=True)
        df_inter.plot(kind="line", figsize=(15,graph_height))

        plt.savefig(self.tmp_png)
        self.excel_chart(sn=sn, start_row=startn, title=f"Imputation {feat}")

        if show_png:
            plt.show()
        plt.close()

        startn += rows_graph +2

        self.excel_add_data(df_inter, sn=sn, start_col=1, start_row = startn, title="Imputation")

    def fake_time_series(self, show_png=False):
        values = [271238, 329285, -1, 260260, 263711]
        timestamps = pd.to_datetime(['2015-01-04 08:29:05',
                                     '2015-01-04 08:34:05',
                                     '2015-01-04 08:39:05',
                                     '2015-01-04 08:44:05',
                                     '2015-01-04 08:49:05'])
        ts = pd.Series(values, index=timestamps)
        ts[ts==-1] = np.nan
        ts = ts.resample('T').mean()

        # https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.08-Multiple-Subplots.ipynb

        fig, axes = plt.subplots(3, 2, figsize=(15,8), sharex='col', sharey='row')

        ts.interpolate(method='spline', order=3).plot(ax=axes[0, 0], title="spline")
        ts.interpolate(method='krogh', order=3).plot(ax=axes[0, 1], title="spline")
        ts.interpolate(method='time').plot(ax=axes[1, 0], title="spline")
        ts.interpolate(method='spline', order=2).plot(ax=axes[1, 1], title="spline")
        ts.interpolate(method='time').plot(ax=axes[2, 0], title="time")
        fig.delaxes(axes[2][1])
        lines, labels = plt.gca().get_legend_handles_labels()
        labels = ['spline', 'time']
        plt.legend(lines, labels, loc='best')

        plt.savefig(self.tmp_png)
        self.excel_chart(sn="TimeSeries", start_row=2, title=f"Time Series Interploation")

        if show_png:
            plt.show()
        plt.close()


    def anomalies(self, df=pd.DataFrame(), per_anomalies=0.02, iid=32, km_iid="id", km_col="sensor21",\
                  max_features=1.0, columns=None, random_state=123, show_png=False):
        """
            three types of anomalies
                1. outliers
                2. change in events
                3. drifts
            methods:
                1. Isolation Forest
                2. luminaire
                2. Local Outlier Factor
                3. Robust Covariance
                4. One-Class SVM
                5. One-Class SVM (SGD)
        """
        ################################################## isolation forests ##################################################
        # del wb['worksheet']._images [1]
        plt.ioff()
        # excel: 1/6 inch heigh. so * 6 for nrows
        if df.empty:
            df = self.train.copy()
        if not columns:
            to_model_columns = list(df)[3:26]

        startn = 2
        sn = "Anomalies"

        graph_height, graph_width = 8, 8

        clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=per_anomalies, \
        max_features=max_features, bootstrap=False, \
        n_jobs=-1, random_state=random_state, verbose=0)

        clf.fit(df[to_model_columns])
        pred = clf.predict(df[to_model_columns])
        df['anomaly']=pred
        outliers=df.loc[df['anomaly']==-1]
        outlier_index=list(outliers.index)

        pca = PCA(n_components=3)  # Reduce to k=3 dimensions
        scaler = StandardScaler()

        X = scaler.fit_transform(df[to_model_columns])
        X_reduce = pca.fit_transform(X)

        fig=plt.figure(figsize=(graph_width,graph_height))

        ax = fig.add_subplot(111, projection='3d')

        ax.set_zlabel("pca_3d")

        # Plot the compressed data points
        ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")
        # Plot x's for the ground truth outliers
        ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
                   lw=2, s=60, marker="x", c="red", label="outliers")
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.tmp_png)
        self.excel_chart(sn, title="Anomalies - Isolation Forest", start_row=startn)
        if show_png:
            plt.show()
        plt.close()

        ################################################## tsne ##################################################
        graph_height, graph_width = 8, 8
        startn += startn+graph_height*6

        df = self.train.copy()
        Y = df.pop("rul")
        Y = [int(y/50) for y in Y]
        scaler = StandardScaler()
        X = scaler.fit_transform(tmp[to_model_columns])

        sns.set(rc={'figure.figsize':(graph_width,graph_height)})
        palette = np.array(sb.color_palette("hls", len(set(Y))))  #Choosing color palette
        x = TSNE(perplexity=30).fit_transform(X)   #perplexity - roughly how many neighbors

        fig = plt.figure(figsize=(graph_width,graph_height))
        ax = fig.add_subplot(111, aspect='equal')
        #ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[Y])

        #txts = []
        xdf = pd.DataFrame(x, columns=["xvalue", "yvalue"])
        xdf["Y"] = Y

        for i in set(Y):
            # Position of each label.
            xtext, ytext = np.median(xdf[xdf["Y"] == 1][["xvalue", "yvalue"]], axis=0)

            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
            #txts.append(txt)

        plt.tight_layout()
        plt.savefig(self.tmp_png)
        self.excel_chart(sn=sn, title="Anomalies - t-SNE visual", start_row=startn)

        if show_png:
            plt.show()
        plt.close()

        ################################################## tsne ##################################################
        if df.empty:
            df = self.train.copy()
        df = df[df[km_iid]==iid]

        graph_height, graph_width = 8, 8
        startn += startn+graph_height*6
        # Construct a Kalman filter
        kf = KalmanFilter(transition_matrices = [1],    # The value for At. It is a random walk so is set to 1.0
                          observation_matrices = [1],   # The value for Ht.
                          initial_state_mean = 0,       # Any initial value. It will converge to the true state value.
                          initial_state_covariance = 1, # Sigma value for the Qt in Equation (1) the Gaussian distribution
                          observation_covariance=1,     # Sigma value for the Rt in Equation (2) the Gaussian distribution
                          transition_covariance=.01)    # A small turbulence in the random walk parameter 1.0
        # Get the Kalman smoothing
        state_means, _ = kf.filter(df[col].values)

        # Call it KF_mean
        df['KF_mean'] = np.array(state_means)
        df.head()

        df[[col,'KF_mean']].plot()
        plt.title(f'Kalman Filter estimates for {col}')
        plt.legend([col,'Kalman Estimate'])
        plt.xlabel('Cycle')
        plt.ylabel('Reading')

        plt.tight_layout()
        plt.savefig(self.tmp_png)
        self.excel_chart(sn=sn, title="Kalman Estimate", start_row=startn)

        if show_png:
            plt.show()
        plt.close()

    def bining(self, df=pd.DataFrame()):
        if df.empty:
            df = self.train.copy()

        sn = "bining"
        startn = 2
        #between
        df["bin_cycle_between"]="D"
        df.loc[df['cycle'].between(0, 50, 'both'), 'bin_cycle_between'] = 'C'
        df.loc[df['cycle'].between(50, 80, 'right'), 'bin_cycle_between'] = 'B'
        df.loc[df['cycle'].between(80, 100, 'right'), 'bin_cycle_between'] = 'A'
        bins = df["bin_cycle_between"].value_counts()
        self.excel_add_data(bins.reset_index(), sn=sn, start_col=1, start_row=startn, title="Bining By pd.between")
        startn += len(bins)+3

        #cut
        df['bin_cycle_cut'] = pd.cut(df['cycle'], bins=[0,30,70,1000], include_lowest=True,right=False, labels=["Low", "Mid", "High"])
        bins = df["bin_cycle_cut"].value_counts()
        self.excel_add_data(bins.reset_index(), sn=sn, start_col=1, start_row=startn, title="Bining by pd.cut")
        startn += len(bins)+3

        #qcut Quantile-based discretization function.Discretize variable into equal-sized buckets based on rank or based on sample quantiles.
        df["bin_cycle_qcut"] = pd.qcut(df["cycle"], 4, labels=False)
        bins = df["bin_cycle_qcut"].value_counts()
        self.excel_add_data(bins.reset_index(), sn=sn, start_col=1, start_row=startn, title="Bining by pd.qcut")
        startn += len(bins)+3

        # for categorical
        list_of_engines = ["turbofan","turboprop","turboshaft","turboprop"]
        conditions = [
            df['Flag'].str.contains('001'),
            df['Flag'].str.contains('002'),
            df['Flag'].str.contains('003'),
            df['Flag'].str.contains('004')]
        df["bin_categorical"] = np.select(conditions, list_of_engines, default='Other')
        bins = df["bin_categorical"].value_counts()
        self.excel_add_data(bins.reset_index(), sn=sn, start_col=1, start_row=startn, title="Categorical Bining")

    def transforming(self, df=pd.DataFrame(), col="sensor21", show_png=False):
        """
            1. Log Transformation / Power Transformer Scaler. (x^lambda-1)/lambda -- when lambda!=0. ln(x) when lambda = 0
            2. MinMax Scaler
            3. Standard Scaler
            4. MaxAbsScaler
            5. Robust Scaler: (x-Q(1)/(Q(3)-Q(1))
            6. Quantile Transformer Scaler: uniform or normal
            7. Unit Vector Scaler/Normalizer
        """
        plt.ioff()
        startn = 2
        show_png = False
        sn = "Transform"
        graph_width, graph_height = 15, 8

        df = self.train.copy()
        df[f"{col}_log"]= df[col].transform(np.log1p) # if with zero: np.log1p(x-x.min()) log1p = log(1+x)

        ###################################################### pt = PowerTransformer() ######################################################
        # pt = PowerTransformer(method="box-cox")
        pt = PowerTransformer(method="yeo-johnson") ##default
        # PowerTranformer methods : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        # yeo-johnson applies to negative as well. box-cox only for positive
        # objectives: stabilize variance and make it normal as possible

        df[[f"{col}_pow"]] = pd.DataFrame(pt.fit_transform(df[[col]].values))
        ## labdas  pt.lambdas_
        pt.lambdas_

        fig, axes = plt.subplots(1,3, figsize=(graph_width,graph_height))

        sns.histplot(data=df, x=f"{col}_log", kde=True, ax=axes[0])
        axes[0].set_title(f"{col}_log")

        sns.histplot(data=df, x=f"{col}", kde=True, ax=axes[1])
        axes[1].set_title(f"{col}")

        sns.histplot(data=df, x=f"{col}_pow", kde=True, ax=axes[2])
        axes[2].set_title(f"{col}_pow")

        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=90)
            ax.set(xlabel=None)
        plt.tight_layout()
        plt.savefig(self.tmp_png)
        self.excel_chart(sn=sn, title="Box-Cox Transformation", start_row=startn)

        if show_png:
            plt.show()
        plt.close()
        startn += graph_height*6 + 2

        # get the lambdas
        df_lambdas = pd.DataFrame({"lambda": pt.lambdas_})
        df_lambdas["Features"] = [col]

        self.excel_add_data(df_lambdas, sn=sn, start_col=1, start_row=startn, title="Lambdas")

        startn += 2

        ###################################################### scaling  ######################################################
        plt.ioff()

        graph_height, graph_width = 8, 15
        df = self.train.copy()
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        df[[f"{col}_robust"]] = scaler.fit_transform(df[[col]].values)

        fig, axes = plt.subplots(1,2, figsize=(graph_width,graph_height))

        sns.histplot(data=df, x=f"{col}_robust", kde=True, ax=axes[0])
        axes[0].set_title(f"{col}_robust")
        sns.histplot(data=df, x=f"{col}", kde=True, ax=axes[1])
        axes[1].set_title(f"{col}")

        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=90)
            ax.set(xlabel=None)

        plt.tight_layout()
        plt.savefig(self.tmp_png)
        self.excel_chart(sn=sn, title="Robust Scaler", start_row=startn)
        if show_png:
            plt.show()
        plt.close()
        startn += graph_height*6 + 2

        # get the lambdas
        df_scale = pd.DataFrame({"Scale": scaler.scale_, "Center":scaler.center_, "Feature": [col] })
        self.excel_add_data(df_scale, sn=sn, start_col=1, start_row=startn, title="Scaling -- Robust Scaling")
        startn += 2

        ###################################################### Tranform  ######################################################
        df = cc.train.copy()
        graph_width, graph_height = 15, 8
        plt.ioff()
        qt = QuantileTransformer(n_quantiles=100, output_distribution='normal') #normal #uniform

        df[[f"{col}_qt"]] = pd.DataFrame(qt.fit_transform(df[[col]].values))

        fig, axes = plt.subplots(1,2, figsize=(15,5))

        sns.histplot(data=df, x=f"{col}_qt", kde=True, ax=axes[0])
        axes[0].set_title(f"{col}_qt")
        sns.histplot(data=df, x=f"{col}", kde=True, ax=axes[1])
        axes[1].set_title(f"{col}")

        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=90)
            ax.set(xlabel=None)

        plt.tight_layout()
        plt.savefig(cc.tmp_png)
        cc.excel_chart(sn=sn, title="Robust Scaler", start_row=startn)

        if show_png:
            plt.show()
        plt.close()

        startn += graph_height*6 + 2

        df_quantitle = pd.DataFrame({"Quantitle":  np.concatenate(qt.quantiles_) })
        df_quantitle["Feature"] = col
        cc.excel_add_data(df_quantitle, sn=sn, start_col=1, start_row=startn, title="Scaling -- Quantile Transformer")
        startn += 2

    def featurization(self,df=pd.DataFrame()):
        """
            method display only

        """
        if df.empty:
            df = self.train.copy()

        ########################################## one hot  ##########################################
        df1 = df.select_dtypes(include="object")

        ohe = OneHotEncoder()
        ohe.fit(df)
        codes_data = ohe.transform(df1).toarray()
        feature_names = ohe.get_feature_names([""])

        tmp = pd.concat([df.select_dtypes(exclude='object'),
           pd.DataFrame(codes_data,columns=feature_names).astype(int)], axis=1)

        ########################################## pd get_dummies  ##################################
        tmp_fd = DF_TRAIN.copy()
        tmp_fd = tmp_fd.select_dtypes(include="object") #string
        tmp_fd = pd.get_dummies(tmp_fd, columns=list(tmp_fd), prefix=["Type_is"] )
        tmp_fd.sample(12)
        # merge with ma

        ########################################## use eval  #####################################
        str_qry = "(Flag=='FD001') & (id>=90) & (id<130) & (cycle<20)"
        tmp = df[df.eval(str_qry)]
        tmp.Flag.str.split("0").map(lambda x: x[-1])

        ########################################## datetime  #####################################
        to_date, from_date = datetime.now(), datetime.now() + timedelta(days=-30)
        df_date = pd.DataFrame({"DT": pd.date_range(start=from_date, end=to_date, freq="H")})

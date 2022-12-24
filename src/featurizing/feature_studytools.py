import os
import pandas as pd
import re
import researchpy as rp
import scipy
import seaborn as sns
import statsmodels.api as sm
import sys
from datetime import datetime, timedelta
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
from IPython.display import display, Markdown, Image
from matplotlib import pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mpl_toolkits.mplot3d import Axes3D
from numpy import isnan
from openpyxl import load_workbook
from pykalman import KalmanFilter
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import pointbiserialr
from scipy.stats import ttest_ind, ttest_1samp, ttest_rel
from sklearn.datasets import make_friedman1
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFdr, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statistics import mode
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import missingno as mn
import numpy as np
import openpyxl

CURRENT_FILE_PATH_PARENT = "c:/weixiuai/src"
sys.path.append(CURRENT_FILE_PATH_PARENT)
from preparing.data_utility import DataUtility
from featurizing.feature_selection_eda import SelectingEDAFeatures
from serving.excel_reporting import ExcelUtility

class FeatureSelection(ExcelUtility, DataUtility):
    """
        C1.1: Anova / f-test
            * Anova uses f-test
            * excercise from scratch: at cycle N, is there any difference among different setting (FD001 - 04)
            * one-way
            * Two types of ANOVA that are commonly used, the one-way ANOVA and the two-way ANOVA
            * independent varialbe: factor(s)
            * dependent: continues numerical measure
            * H0: no difference between groups and means
            * H1: there is difference
            * Normality | Sample independence | Variance equality | continuous dependent variable
        C1.2: t-test
            * In statistics, t-tests are a type of hypothesis test that allows you to compare means.
            * There are three t-tests to compare means: a one-sample t-test, a two-sample t-test and a paired t-test.
              one sample t-statistic = (x_bar - mu)/(stdev/sqrt(n))
        C1.3. Correlations
            * four types of correlations:  Pearson correlation, Kendall rank correlation, Spearman correlation, and the Point-Biserial correlation
        C1.4: Information Theory Basics
            * MI. Mutual Information
            * IG. Information Gain, Entropy reduction.
            * Entropy. In information theory, the entropy of a random variable is the average level of "information", "surprise", or "uncertainty" inherent to the variable's possible outcomes.
            * KL. Kullback-Leibler divergence.  a measure that calculates the difference between two probability distributions.
            * Calculate Entropy
                $ E = - \sum\limits_{i=0}^N P_ilog_2P_i $
                $P_i $ is probability of randomly select one being class i. total N classes

                * entropy = 0: pure
                * entropy the higher the messier

    """
    def __init__(self,  flags=None, excel_file_prefix=None):
        if not flags:
            flags = ["FD001","FD002","FD003","FD004"]
        if not excel_file_prefix:
            excel_file_prefix = "FeatureResearch"
        ExcelUtility.__init__(self, excel_file_prefix=excel_file_prefix)
        DataUtility.__init__(self)
        self.train, self.test, self.result = self.load_data_by_flags(flags=flags)
        self.prepare_excel_report()

    def get_data(self, df=pd.DataFrame(),  qry_str=None):
        if df.empty:
            df = self.train.copy()

        if not qry_str:
            measure = "sensor21"
            cycle = 12
            factor = "Flag"
            qry_str = f"cycle=={cycle}"

        df = df[df.eval(qry_str)][[factor, measure]]

        return df

    def t_test(self,  df=pd.DataFrame(),  qry_str=None):
        """
            # use researchpy ttest. https://github.com/researchpy/researchpy/tree/master/researchpy
            # st.ttest_1samp(a=vals, popmean=23.36, alternative = "less") #"alternative" for two-sided, less, greater
        """
        df = self.get_data(df=df, qry_str=qry_str)

        #### researchpy ttest
        summary, results = rp.ttest(group1= df['sensor21'][df['Flag'] == 'FD001'], group1_name= "FD001",
                                    group2= df['sensor21'][df['Flag'] == 'FD002'], group2_name= "FD002")
        # it also measures Point-Biserial r - A point-biserial correlation is used to measure the strength
        #and direction of the association that exists between one continuous variable and one dichotomous variable.

        # one sample
        vals = df["sensor21"].values
        st.ttest_1samp(a=vals, popmean=15.75, alternative = "less") #"alternative" for two-sided, less, greater

        # ttest_ind
        group1= df['sensor21'][df['Flag'] == 'FD001']
        group2= df['sensor21'][df['Flag'] == 'FD003']
        ttest_ind(a=group1, b=group2, equal_var=True)

        # paired
        group1 = df[df["Flag"] == "FD001"]['sensor21'][:20]
        group2 = df[df["Flag"] == "FD002"]['sensor21'][:20]
        ttest_rel(a=group1, b=group2 )

    def f_test_anova(self,df=pd.DataFrame(),  qry_str=None):
        df = self.get_data(df=df, qry_str=qry_str)
        group1= df['sensor21'][df['Flag'] == 'FD001']
        group2= df['sensor21'][df['Flag'] == 'FD003']
        f = np.var(group1, ddof=1)/np.var(group2, ddof=1) #calculate F test statistic

        dfn = len(group1)-1 #define degrees of freedom numerator
        dfd = len(group2)-1 #define degrees of freedom denominator
        p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic

        group1= df['sensor21'][df['Flag'] == 'FD001']
        group2= df['sensor21'][df['Flag'] == 'FD003']
        f = np.var(group1, ddof=1)/np.var(group2, ddof=1) #calculate F test statistic

        fvalue, pvalue = stats.f_oneway(group1,group2)

        #more groups:
        fvalue, pvalue = stats.f_oneway(cc.train['sensor21'], cc.train['sensor20'], cc.train['sensor11'])


    def plot_f_distr(self, x_range, dfn, dfd, mu=0, sigma=1, cdf=False, **kwargs):
        '''
        Plots the f distribution function for a given x range, dfn and dfd
        If mu and sigma are not provided, standard f is plotted
        If cdf=True cumulative distribution is plotted
        Passes any keyword arguments to matplotlib plot function
        '''
        x = x_range
        if cdf:
            y = scipy.stats.f.cdf(x, dfn, dfd, mu, sigma)
        else:
            y = scipy.stats.f.pdf(x, dfn, dfd, mu, sigma)

        plt.plot(x, y, **kwargs)
        return y

    def f_test_plot(self, df=pd.DataFrame(), N=5, sensor="sensor21"):

        if df.empty:
            df = self.train.copy()

        qry_str = f"(Flag=='FD001' | Flag=='FD002') & (rul=={N})"
        df = df[df.eval(qry_str)][["Flag", sensor]]

        set_x = df[df["Flag"]=="FD001"][sensor]
        set_y = df[df["Flag"]=="FD002"][sensor]

        dn_fd001 = len(set_x)
        dn_fd002 = len(set_y)

        x = np.linspace(0.0, 5, 200)
        increment = 5/200
        alpha = 0.05

        dn_fd001= len(set_x) - 1
        dn_fd002= len(set_y) - 1

        self.plot_f_distr(x, dn_fd001, dn_fd002, 0, 1, color='red', lw=2, ls='-', alpha=0.5, label='pdf')
        y = self.plot_f_distr(x, dn_fd001, dn_fd002, 0, 1, cdf=True, color='blue', lw=2, ls='-', alpha=0.5, label='cdf')

        start = 0
        for i, value in enumerate(y):
            start = increment*i
            if value > 1-alpha:
                break

        px=np.arange(start,5,0.1)
        plt.fill_between(px, scipy.stats.f.pdf(px, dn_fd001, dn_fd002, 0, 1),color='g')

        plt.title(f"f-distribution with {dn_fd001} degree of freedoms")

        plt.legend();

    def anova_one_way(self, df=pd.DataFrame(), qry_str=None):
        """
            from scratch
        """
        if df.empty:
            df = self.train.copy()
        alpha, cycle, factor,measure = 0.05, 23, "Flag", 'sensor21'
        if not qry_str:
            qry_str = f"rul=={cycle}"
        df = df[df.eval(qry_str)][["Flag", measure]]

        ## first boxplot it
        fig = plt.figure(figsize=(15,5))
        sns.set_style("white")
        box_plot = sns.boxplot(x=factor,y=measure,data=df)

        medians = df.groupby([factor])[measure].median()
        vertical_offset = df[measure].median() * 0.05 # offset from median for display

        for xtick in box_plot.get_xticks():
            box_plot.text(xtick,medians[xtick].round(2) + vertical_offset,medians[xtick].round(2),
                    horizontalalignment='center',size='large',color='w',weight='semibold')

        overall_mean = df[measure].mean()
        # compute Sum of Squares Total
        df['overall_mean'] = overall_mean
        ss_total = sum((df[measure] - df['overall_mean'])**2)

        # compute group means
        group_means = df.groupby(factor).mean()
        group_means = group_means.rename(columns = {measure: 'group_mean'})

        df = df.merge(group_means[["group_mean"]], left_on = factor, right_index = True)
        ss_residual = sum((df[measure] - df['group_mean'])**2)
        ss_explained = sum((df['overall_mean'] - df['group_mean'])**2)


        # compute Mean Square Residual
        n_groups = len(set(df[factor]))
        n_obs = df.shape[0]
        df_residual = n_obs - n_groups
        ms_residual = ss_residual / df_residual

        # compute Mean Square Explained
        df_explained = n_groups - 1
        ms_explained = ss_explained / df_explained

        # compute F-Value
        f = ms_explained / ms_residual

        # compute p-value
        p_value = 1 - scipy.stats.f.cdf(f, df_explained, df_residual)
        if p_value <= alpha:
            display(Markdown(f"H0 Rejected with p_value as {p_value.round(5)}</font>\n\n"))
        else:
            display(Markdown(f"H0 Stands with p_value as {p_value.round(5)}</font>\n\n"))

    def chi2(self, df=pd.DataFrame()):
        """
            test of independence. The  𝜒2  test is one of the statistical tests we can use to decide whether
            there is a correlation between the categorical variables by analysing the
            relationship between the observed and expected values
            𝜒2=∑(𝑂𝑖−𝐸𝑖)2/𝐸𝑖

            Yates adjustments:
            𝜒2𝑌𝑎𝑡𝑒𝑠=∑(|𝑂𝑖−𝐸𝑖|−0.5)2𝐸𝑖
        """
        # goal: find is there any relationship between Op/Fault setting and Life
        if df.empty:
            df = self.train.copy()

        df = df.groupby(["Flag", "id"])["cycle"].max().reset_index()
        df["Life"] =  pd.cut(df['cycle'], bins=[0,200,300, 400, 500, 600], include_lowest=True,right=False, labels=["Low", "Mid-Low", "Mid", "Mid-High", "High"])
        df.groupby("Life")["id"].count().reset_index()
        contingency_table = pd.crosstab(df["Life"], df["Flag"])

        # use chisquare
        results = scipy.stats.chisquare(contingency_table, f_exp=None)
        results.statistic, results.pvalue # 1-D array

    def correlation(self, df=pd.DataFrame()):
        """
            four types of correlations: Pearson correlation, Kendall rank correlation, Spearman correlation, and the Point-Biserial correlation
            1. pearson
                It is the ratio between the covariance of two variables and the product of their standard deviations
                𝑟=∑(𝑥−𝑥¯)(𝑦−𝑦¯)∑(𝑥−𝑥¯)2∑(𝑦−𝑦¯)2√
            2. vif -- The variance inflation factor is a measure for the increase of the variance of the parameter estimates if an additional variable,
                given by exog_idx is added to the linear regression. It is a measure for multicollinearity of the design matrix, exog.
            3. Kendall -- Kendall rank correlation is a non-parametric test that measures the strength of dependence between two variables. If we consider two samples, a and b, where each sample size is n, we know that the total number of pairings with a b is n(n-1)/2. The following formula is used to calculate the value of Kendall rank correlation:
                𝜏=𝑛𝑐−𝑛𝑑12𝑛(𝑛−1)

                𝑛𝑐 : number of concordant
                𝑛𝑑 : number of discordant

            4. spearman -   Spearman rank correlation
                Spearman rank correlation is a non-parametric test that is used to measure the degree of association between two variables.
                The following formula is used to calculate the Spearman rank correlation:
                𝜌=1−6∑𝑑2𝑖𝑛(𝑛2−1)

                𝜌 : spearman rank correlation
                𝑑𝑖  the difference between the ranks of correspondent variables
                n: number of observations
                Kendall and Spearman are mostly interchangeable. Kendall is more robust

            5. Point-Biserial correlation
                The Point-Biserial Correlation Coefficient is a correlation measure of the strength of association between a
                continuous-level variable (ratio or interval data) and a binary variable. Binary variables are variables of nominal
                scale with only two values.
                classical usage: test whether a multiple choise test is too easy or too hard.

        """
        # pearson
        if df.empty:
            df = self.train.copy()

        kept_columns = [c for c in list(df) if re.search("sens", c)]

        tmp_corr = df[kept_columns].corr(method="pearson")
        plt.figure(figsize=(9, 9))
        ax = sns.heatmap(tmp_corr,  square=True, annot=True,annot_kws={"size": 8},
                    center=0, fmt=".2f",   linewidths=.5,  #fmt=".2g",
                    cmap="vlag", cbar_kws={"shrink": 0.8});
        ax.xaxis.tick_top()
        plt.xticks(rotation = 45)
        plt.show()


        # kendall's tau and spearman: numerical or ordinal. two variables
        # Spearman’s Rho and Kendall’s Tau are very similar tests and are used in similar scenarios.
        # We (Scipy) recommend using Kendall’s Tau first and Spearman’s Rho as a backup.
        # z = 3τ*√n(n-1) / √2(2n+5) the p-value in kendall is for the z-value
        # spearman rho: Greek letter {\displaystyle \rho }\rho  (rho)
        # The point biserial correlation is used to measure the relationship between a binary variable, x, and a continuous variable, y

        if df.empty:
            df = self.train.copy()

        tau, p_value = scipy.stats.kendalltau(df["sensor2"], df["sensor3"])
        rho, p_spearman = scipy.stats.spearmanr(df["sensor2"], df["sensor3"])
        tau, p_value, rho, p_spearman

        df = df[df["Flag"].isin(["FD001", "FD002"])]
        df["Flag2"] = np.where(df["Flag"]=="FD001", 1, 0)
        pbc = pointbiserialr(df["sensor21"], df["Flag2"])
        # PointbiserialrResult(correlation=0.6932332897269851, pvalue=0.0) pvalue is the significance of corr

    def pearson_cor_selector(self, df=pd.DataFrame(), col="rul", flag="FD001", num_feats = 10):
        if df.empty:
            df=self.train.copy()
        qry_str = f"Flag=='{flag}'"

        tmp = df.copy()
        tmp = tmp[tmp.eval(qry_str)]

        tmp.drop(columns=["id", "cycle", "Flag"], axis=1, inplace=True)

        y = tmp.pop("rul")
        X = tmp
        cor_list = []
        feature_name = X.columns.tolist()
        # calculate the correlation with y for each feature
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)
        # replace NaN with 0
        cor_list = [0 if np.isnan(i) else i for i in cor_list]

        # feature name
        cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()

        # feature selection? 0 for not select, 1 for select
        cor_support = [True if i in cor_feature else False for i in feature_name]
        col_indexs = np.argsort(np.abs(cor_list))[-num_feats:]

        return cor_support, cor_feature, cor_list, col_indexs

    def corr_vif(self, df=pd.DataFrame()):

        if df.empty:
            df = self.train.copy()

        df = df[df["Flag"]=="FD001"]
        df = df.select_dtypes(include="number")
        df.drop(df = ["id", "cycle"], inplace = True)

        vif_data = pd.DataFrame()
        vif_data["feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
        vif_data.round(2)

    def entropy_mi_kl(self, df=pd.DataFrame()):
        """
            Entropy: 𝐸=−∑𝑖=0𝑁𝑃𝑖𝑙𝑜𝑔2𝑃𝑖
                entropy = 0: pure
                entropy the higher the messier
            MI/KL: Mutual information is calculated between two variables and measures the reduction in uncertainty for
                one variable given a known value of the other variable.
                Kullback-Leibler, or KL, divergence is a measure that calculates the difference between two probability distributions.
                𝐼(𝑋;𝑌)=𝐻(𝑋)−𝐻(𝑋|𝑌)
                Where I(X ; Y) is the mutual information for X and Y, H(X) is the entropy for X and H(X | Y) is the conditional entropy for X given Y. The result has the units of bits.

                Sometime it is also expressed in KL:
                𝐼(𝑋;𝑌)=𝐾𝐿(𝑝(𝑋,𝑌)||𝑝(𝑋)∗𝑝(𝑌))
                they are not the same.
                calculate KL
                𝐾𝐿(𝑃||𝑄)=∑𝑝𝑖(𝑥)𝑙𝑜𝑔(𝑝𝑖(𝑥)𝑞𝑖(𝑥))
                if continuous:
                𝐾𝐿(𝑃||𝑄)=∫𝑝(𝑥)𝑙𝑜𝑔𝑝(𝑥)𝑞(𝑥)
        """
        if df.empty:
            df = self.train.copy()

        # sklearn implementation
        df["Life"] =  pd.cut(df['cycle'], bins=[0,200,300, 400, 500, 600], include_lowest=True,right=False, \
                             labels=["Low", "Mid-Low", "Mid", "Mid-High", "High"])
        y = df.pop('Flag')

        tmp_df = pd.get_dummies(df[["Life"]], columns=["Life"], prefix=["Stage_"] )
        mic_score = MIC(tmp_df.values,y,n_neighbors=3, discrete_features = True) # it is an estimate

        #print(mic_score, sum(mic_score))
        # It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
        return mic_score, sum(mic_score)

    def feature_filter(self, df=pd.DataFrame()):
        tmp = DF_TRAIN.copy()
        tmp["rul"] = tmp.groupby(["id", "Flag"])["cycle"].transform("max") - tmp["cycle"]

        fault_op = "FD001"
        qry_str = f"Flag=='{fault_op}'"

        tmp = tmp[tmp.eval(qry_str)]
        tmp["Distance_To_Fail"] =  pd.cut(tmp['rul'], bins=[0,20,50, 100, 200, 600], include_lowest=True,right=True, labels=["Low", "Mid-Low", "Mid", "Mid-High", "High"])

            tmp_clf = tmp.copy()
        y = tmp_clf.pop("Distance_To_Fail")
        tmp_clf.drop(columns=["id", "cycle", "Flag", "rul"], inplace=True)

        # important: none negative
        tmp_clf[tmp_clf < 0] = 0

        # select the two best features
        kb = SelectKBest(chi2, k=10)
        X_new = kb.fit_transform(tmp_clf, y)

        kb.get_feature_names_out()


        # select the two best features
        kb = SelectKBest(f_regression, k=10)
        X_new = kb.fit_transform(tmp_clf, y)

        kb.get_feature_names_out()

        fdr = SelectFdr(score_func=f_regression, alpha=0.05)

        tmp_clf = tmp.copy()
        y = tmp_clf.pop("rul")
        tmp_clf.drop(columns=["id", "cycle", "Flag", "Distance_To_Fail"], inplace=True)

        # important: none negative
        tmp_clf[tmp_clf < 0] = 0
        X_new = fdr.fit_transform(tmp_clf, y)
        fdr.get_feature_names_out()

    def feature_wrapper(self):
        tmp_clf = tmp.copy()
        y = tmp_clf.pop("rul")
        tmp_clf.drop(columns=["id", "cycle", "Flag", "Distance_To_Fail"], inplace=True)

        # important: none negative
        tmp_clf[tmp_clf < 0] = 0

        sfs1 = SFS(RandomForestRegressor(),
               k_features=10,
               forward=True,
               floating=False,
               verbose=2,
               scoring='r2',
               cv=3)

        sfs1 = sfs1.fit(tmp_clf.values, y)
        sfs1.k_feature_idx_
        tmp_clf.columns[list(sfs1.k_feature_idx_)]
        # step backward feature elimination

        tmp_clf = tmp.copy()
        y = tmp_clf.pop("rul")
        tmp_clf.drop(columns=["id", "cycle", "Flag", "Distance_To_Fail"], inplace=True)

        # important: none negative
        tmp_clf[tmp_clf < 0] = 0

        sfs2 = SFS(RandomForestRegressor(),
               k_features=10,
               forward=False,
               floating=False,
               verbose=2,
               scoring='r2',
               cv=3)

        sfs2 = sfs2.fit(tmp_clf.values, y)

        # RFECV(estimator, *, step=1, min_features_to_select=1, cv=None, scoring=None, verbose=0, n_jobs=None, importance_getter='auto')

        tmp_clf = tmp.copy()
        y = tmp_clf.pop("rul")
        tmp_clf.drop(columns=["id", "cycle", "Flag", "Distance_To_Fail"], inplace=True)

        # important: none negative
        tmp_clf[tmp_clf < 0] = 0
        estimator = SVR(kernel="linear")

        selector = RFECV(estimator, step=1, cv=5)
        selector = selector.fit(tmp_clf, y)
        selector.support_
        selector.ranking_

    def feature_embedding(self):
        tmp_clf = tmp.copy()
        y = tmp_clf.pop("rul")
        tmp_clf.drop(columns=["id", "cycle", "Flag", "Distance_To_Fail"], inplace=True)
        scaler = StandardScaler()
        scaler.fit(X_train.fillna(0))

        sel_ = SelectFromModel(Lasso(alpha=100))
        sel_.fit(scaler.transform(tmp_clf.fillna(0)), y)

class LassoRegression() :
    def __init__( self, learning_rate, iterations, l1_penality ) :
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penality = l1_penality
    # Function for model training

    def fit( self, X, Y ) :
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
        # weight initialization
        self.W = np.zeros( self.n )
        self.b = 0
        self.X = X
        self.Y = Y
        # gradient descent learning
        for i in range( self.iterations ) :
            self.update_weights()
        return self

    # Helper function to update weights in gradient descent
    def update_weights( self ) :
        Y_pred = self.predict( self.X )
        # calculate gradients
        dW = np.zeros( self.n )
        for j in range( self.n ) :
            if self.W[j] > 0 :
                dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) )
                         + self.l1_penality ) / self.m
            else :
                dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) )
                         - self.l1_penality ) / self.m
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m
        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self
    # Hypothetical function  h( x )
    def predict( self, X ) :
        return X.dot( self.W ) + self.b

def main() :
    # Importing dataset
    df = pd.read_csv( "salary_data.csv" )
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values
    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1 / 3, random_state = 0 )
    # Model training
    model = LassoRegression( iterations = 1000, learning_rate = 0.01, l1_penality = 500 )
    model.fit( X_train, Y_train )
    # Prediction on test set
    Y_pred = model.predict( X_test )
    print( "Predicted values ", np.round( Y_pred[:3], 2 ) )
    print( "Real values      ", Y_test[:3] )
    print( "Trained W        ", round( model.W[0], 2 ) )
    print( "Trained b        ", round( model.b, 2 ) )
    # Visualization on test set
    plt.scatter( X_test, Y_test, color = 'blue' )
    plt.plot( X_test, Y_pred, color = 'orange' )
    plt.title( 'Salary vs Experience' )
    plt.xlabel( 'Years of Experience' )
    plt.ylabel( 'Salary' )
    plt.show()

class RidgeRegression() :
    def __init__( self, learning_rate, iterations, l2_penality ) :

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_penality = l2_penality

    # Function for model training
    def fit( self, X, Y ) :

        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape

        # weight initialization
        self.W = np.zeros( self.n )

        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning
        for i in range( self.iterations ) :
            self.update_weights()
        return self

    # Helper function to update weights in gradient descent

    def update_weights( self ) :
        Y_pred = self.predict( self.X )

        # calculate gradients
        dW = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) +
               ( 2 * self.l2_penality * self.W ) ) / self.m
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    # Hypothetical function  h( x )
    def predict( self, X ) :
        return X.dot( self.W ) + self.b

    def get_ridge(self):
        df = pd.read_csv( "salary_data.csv" )
        X = df.iloc[:, :-1].values
        Y = df.iloc[:, 1].values

        # Splitting dataset into train and test set
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y,

                                              test_size = 1 / 3, random_state = 0 )

        # Model training
        model = RidgeRegression( iterations = 1000,
                                learning_rate = 0.01, l2_penality = 1 )
        model.fit( X_train, Y_train )

        # Prediction on test set
        Y_pred = model.predict( X_test )
        print( "Predicted values ", np.round( Y_pred[:3], 2 ) )
        print( "Real values      ", Y_test[:3] )
        print( "Trained W        ", round( model.W[0], 2 ) )
        print( "Trained b        ", round( model.b, 2 ) )

        # Visualization on test set
        plt.scatter( X_test, Y_test, color = 'blue' )
        plt.plot( X_test, Y_pred, color = 'orange' )
        plt.title( 'Salary vs Experience' )
        plt.xlabel( 'Years of Experience' )
        plt.ylabel( 'Salary' )
        plt.show()

cc = FeatureSelection()

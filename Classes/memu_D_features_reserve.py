
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR 
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler 

class FeatureSelection(object):
    def __init__(self, random_state=123, test_size=0.3, features = None):
        """
            documentation: 
            this module will be explained and utilized later for Classification
        """ 
        self.test_size = test_size
        self.random_state = random_state 
        self.features = features
        
    def split_df(self, df): 
        """
            doc: 
            this module will be explained and utilized later for Classification
        """ 
        Y = df["rul"]
        X = df[self.feats]
        return train_test_split(X,Y, test_size=self.test_size, random_state=self.random_state)

    def select_features(self, df, to_plot=True):  
        """
            doc: 
            this module will be explained and utilized later for Classification
        """ 
        fs = SelectKBest(score_func=f_regression, k="all")
        # learn relationship from training data
        fs.fit(X_train, y_train)
        # transform train input data
        X_train_fs = fs.transform(X_train)
        # transform test input data
        X_test_fs = fs.transform(X_test)
        
        if to_plot: 
            # what are scores for the features
            for i in range(len(fs.scores_)):
                print('Feature %d: %f' % (i, fs.scores_[i]))
            # plot the scores
            sns.barplot([i for i in range(len(fs.scores_))], fs.scores_) 
            plt.show() 
        return X_train_fs, X_test_fs, fs
    
    def wrapping(self, df, y):  
        """
            doc: 
            this module will be explained and utilized later for Classification
        """ 
        estimator = SVR(kernel="linear") 
        selector = RFECV(estimator, step=1, cv=5)
        selector = selector.fit(df, y)
        #selector.support_ 
        #selector.ranking_ 
        return selector

    def embedded(self, df): 
        scaler = StandardScaler()
        scaler.fit(tmp) 
        sel_ = SelectFromModel(Lasso(alpha=1)) # lambda 
        sel_.fit(scaler.transform(df), y) 
        #sel_.estimator_.coef_ 
        #sel_.threshold_
        #sel_.get_support() 
        return sel_

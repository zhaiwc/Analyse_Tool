# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:50:29 2018

@author: zhaiweichen
"""
import pandas as pd
import numpy as np
import pdb

import sklearn.ensemble as esb
from sklearn.decomposition import PCA,FactorAnalysis,KernelPCA,SparsePCA,TruncatedSVD,IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import  GridSearchCV,cross_val_score
from sklearn.feature_selection import SelectKBest,VarianceThreshold
from sklearn.metrics import make_scorer,mean_squared_error,roc_auc_score, accuracy_score,mutual_info_score
from sklearn import linear_model

from xgboost import XGBRegressor,XGBClassifier
from Analysis_Tool import Data_plot,Data_Preprocess

class Feature_Reduction():
    '''
    n_comp: N 前N大主成分（优先）
    cum_std ： 0.8 前N大主成分所解释的方差累计百分比（次优）
    '''
    def __init__(self, n_comp  ,method = 'PCA',  cum_std = 0.8):
        self.method = method
        self.n_comp = n_comp
        self.cum_std = cum_std
        self.dr_model = None
        self.data_col = None
        
    def fit(self,data):
        '''
        训练数据
        '''
        #先以n_comp为基准，如不满足条件则
        if self.method == 'pca':
            self.dr_model = PCA(n_components = self.n_comp)
            self.dr_model.fit(data)
            if self.dr_model.explained_variance_ratio_.cumsum()[-1] < self.cum_std:
                self.dr_model = PCA(n_components = self.cum_std)
                self.dr_model.fit(data)
                
        elif self.method == 'kpca':
            self.dr_model = KernelPCA(n_components = self.n_comp,kernel="rbf")
            self.dr_model.fit(data)
            
        elif self.method == 'fa':
            self.dr_model = FactorAnalysis(n_components = self.n_comp)
            self.dr_model.fit(data)
        
        elif self.method == 'spca':
            self.dr_model = SparsePCA(n_components = self.n_comp)
            self.dr_model.fit(data)
            
        elif self.method == 'tsvd':
            self.dr_model = TruncatedSVD(n_components = self.n_comp)
            self.dr_model.fit(data)
        
        elif self.method == 'ipca':
            self.dr_model = IncrementalPCA(n_components = self.n_comp)
            self.dr_model.fit(data)
            
        self.data_col = data.columns
        
    def transform(self,data):
        '''
        对预测数据进行转换
        '''
        if self.dr_model is not None:
            if self.method == 'pca':
                res = self.dr_model.transform(data)
                col = ['pca'+ str(i+1) for i in range(res.shape[1])]
                res = pd.DataFrame(res,columns = col)
                
            elif self.method == 'kpca':
                res = self.dr_model.transform(data)
                col = ['KPCA'+ str(i+1) for i in range(res.shape[1])]
                res = pd.DataFrame(res,columns = col)
            
            elif self.method == 'fa':
                res = self.dr_model.transform(data)
                col = ['FA'+ str(i+1) for i in range(res.shape[1])]
                res = pd.DataFrame(res,columns = col)
                
            elif self.method == 'spca':
                res = self.dr_model.transform(data)
                col = ['SPCA'+ str(i+1) for i in range(res.shape[1])]
                res = pd.DataFrame(res,columns = col)
            
            elif self.method == 'tsvd':
                res = self.dr_model.transform(data)
                col = ['TSVD'+ str(i+1) for i in range(res.shape[1])]
                res = pd.DataFrame(res,columns = col)
                
            elif self.method == 'ipca':
                res = self.dr_model.transform(data)
                col = ['ipca'+ str(i+1) for i in range(res.shape[1])]
                res = pd.DataFrame(res,columns = col)
        else:
            print('dr_model is None')
            res = None
        return res
    
    def get_vip(self,wight_comp,Top_N = 20 ):
        '''
        wight_comp:由其他模型计算得到的因子重要性
        根据wight_comp * transformmat 计算原始因子重要性
        '''
        if self.method in ['pca','fa','spca','tsvd','ipca']:
            weight = np.array(wight_comp).reshape(1,-1)
            trans_mat = self.dr_model.components_
#            pdb.set_trace()
            res = pd.DataFrame(np.dot(weight,trans_mat),columns = self.data_col,index = ['importance'])
            res = abs(res.T).sort_values('importance')
            return res.iloc[-Top_N:]
        else:
            return None
    
    def plot_score_scatter(self,data,label_col = None,is_plot = True):
        dr_data = self.dr_model.transform(data)
        plot_data = pd.DataFrame(dr_data[:,0:2],index = data.index,columns = ['pca1','pca2'])

              
        #画x,y轴
        max_x = max(dr_data[:,0].std()*3,dr_data[:,0].max())
        max_y = max(dr_data[:,1].std()*3,dr_data[:,1].max())
        x_mat = pd.DataFrame([[max_x,0],[-max_x,0]])
        y_mat = pd.DataFrame([[0,-max_y],[0,max_y]])
        if is_plot:
            plt = Data_plot.plot_scatter(plot_data,label_col = label_col,issns = False)  
            plt = Data_plot.plot_line(x_mat,c=['b--'])
            plt = Data_plot.plot_line(y_mat,c=['b--'])
            plt.xlabel('pca1')
            plt.ylabel('pca2')
            plt.title('pca1 vs pca2 scatter')
            plt.show()
        return plot_data
    
    def plot_loading_scatter(self,data,label_col = None,is_plot = True):
        if self.method in ['pca','fa','spca','tsvd','ipca']:
            comp_mat = self.dr_model.components_.T[:,0:2]
            plot_data = pd.DataFrame(comp_mat,index = data.columns,columns = ['comp1','comp2'])
             
             #画x,y轴
            max_x = max(comp_mat[:,0].std()*3,comp_mat[:,0].max())
            max_y = max(comp_mat[:,1].std()*3,comp_mat[:,1].max())
            x_mat = pd.DataFrame([[max_x,0],[-max_x,0]])
            y_mat = pd.DataFrame([[0,-max_y],[0,max_y]])
            
            if is_plot:
                plt = Data_plot.plot_scatter(plot_data,label_col = label_col,issns = False)
                plt = Data_plot.plot_line(x_mat,c=['b--'])
                plt = Data_plot.plot_line(y_mat,c=['b--'])
                plt.xlabel('pca1')
                plt.ylabel('pca2')
                plt.title('pca1 vs pca2 loadings')
                plt.show()
            return plot_data
        else:
            return None
        
    def plot_cum_std(self,data,n = 30):
        '''
        画各个维度pca累积贡献度
        '''
        pca_model = PCA(n_components = n)
        pca_model.fit(data)
        eplan_var_csum = pca_model.explained_variance_ratio_.cumsum()
        plt = Data_plot.plot_line(pd.DataFrame(eplan_var_csum,columns = ['cum_var']))
#        plt = Data_plot.plot_line(pd.DataFrame(np.ones(eplan_var_csum.shape)*0.8,columns=['base_line']))
        plt.title('cumsum explained_variance_ratio')
        plt.show()
        
class Filter_Selection():
    def __init__(self,method,TopN = 10):
        self.method = method
        self.TopN = TopN
        self.filter = None
        self.choose_col = []
        
    def fit(self,x,y):
        if self.method == 'std':
            #标准差
            std_x = np.std(x).sort_values()
            self.choose_col = std_x.index[-self.TopN:]
            
        elif self.method == 'pearson':
            #皮尔森相关系数
            corr_x = abs(pd.concat([x,y],axis = 1).corr().iloc[:-1,-1]).sort_values()
            self.choose_col = corr_x.index[-self.TopN:]
            
        elif self.method == 'mi':
            #互信息
            mi = []
            for col in x.columns:
                temp_x = x.loc[:,[col]].values.reshape(len(y),)
                temp_y = y.values.reshape(len(y),)
                mi.append(mutual_info_score(temp_x,temp_y))
            mi = pd.DataFrame(mi,index = x.columns,columns = ['mi']).sort_values('mi')
            self.choose_col = mi.index[-self.TopN:]
        
            
    def transform(self,x):
        transform_x = x.loc[:,self.choose_col]
        return transform_x

class Reg_Embedded_Selection():
    def __init__(self,method,TopN = 10):
        self.method = method
        self.parameters = None
        self.select_col = None
        self.Top_N = TopN
        
    def set_parameters(self,parameters = None):
        if parameters is None: #用户不传参数，则使用默认参数
            if self.method == 'ElasticNet':
                self.parameters = {'alpha':[1,],
                                   "l1_ratio":[0.5,]}
                
            elif self.method == 'rf':
                self.parameters = {"max_depth": [5],
                        "n_estimators": [200],}
                
            elif self.method == 'adaBoost':
                self.parameters = {'n_estimators':[200],
                                   'learning_rate':[0.1],}
                
            elif self.method == 'gbm':
                self.parameters = {"max_depth": [5],
                                   "learning_rate": [0.1],
                                   "n_estimators": [200],}
                
            elif self.method == 'xgb':
                self.parameters = { "max_depth": [5],
                                   "learning_rate": [0.1],
                                   "n_estimators": [200],}
        else:
            self.parameters = parameters
    
    def fit(self,x,y):
        
        x_col = x.columns
        scoring = {"mse": make_scorer(mean_squared_error),}
        
        x = x.values
        y = y.values.reshape(len(y),)
        
        if self.parameters is None:
            self.set_parameters()
            
        if self.method == 'ElasticNet':
            reg = GridSearchCV(linear_model.ElasticNet(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            reg.fit(x,y)
            coef = pd.DataFrame(reg.best_estimator_.coef_,index = x_col,columns = ['coefs']).sort_values('coefs')
            self.select_col = coef.index[-self.TopN:]
        
        elif self.method == 'rf':
            reg = GridSearchCV(esb.RandomForestRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            reg.fit(x,y)
            
            importances = reg.best_estimator_.feature_importances_
            Rank = pd.DataFrame(importances,index=x_col ,columns = ['importances']).sort_values('importances')
            self.select_col = list(Rank.index[-self.Top_N:])
        
        elif self.method == 'adaBoost':
            reg = GridSearchCV(esb.AdaBoostRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            reg.fit(x,y)
            
            importances = reg.best_estimator_.feature_importances_
            Rank = pd.DataFrame(importances,index=x_col,columns = ['importances']).sort_values('importances')
            self.select_col = list(Rank.index[-self.Top_N:])
        
        elif self.method == 'gbm':
            reg = GridSearchCV(esb.GradientBoostingRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            reg.fit(x,y)
            
            importances = reg.best_estimator_.feature_importances_
            Rank = pd.DataFrame(importances,index=x_col,columns = ['importances']).sort_values('importances')
            self.select_col = list(Rank.index[-self.Top_N:])
            
        elif self.method == 'xgb':
            reg = GridSearchCV(XGBRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            reg.fit(x,y)
            
            importances = reg.best_estimator_.feature_importances_
            Rank = pd.DataFrame(importances,index=x_col,columns = ['importances']).sort_values('importances')
            self.select_col = list(Rank.index[-self.Top_N:])
            
            
    def transform(self,x):
        transform_x = x.loc[:,self.select_col]
        return transform_x
    
class Cls_Embedded_Selection():
    def __init__(self,method,TopN = 10):
        self.method = method
        self.parameters = None
        self.select_col = None
        self.TopN = TopN
        
    def set_parameters(self,parameters = None):
        if parameters is None: #用户不传参数，则使用默认参数
            if self.method == 'logistic':
                self.parameters = {'penalty':['l1',],'C':[1]}
                
            elif self.method == 'rf':
                self.parameters = {"max_depth": [5],
                        "n_estimators": [200],}
                
            elif self.method == 'adaBoost':
                self.parameters = {'n_estimators':[200],
                                   'learning_rate':[0.1],}
                
            elif self.method == 'gbm':
                self.parameters = {"max_depth": [5],
                                   "learning_rate": [0.1],
                                   "n_estimators": [200],}
                
            elif self.method == 'xgb':
                self.parameters = { "max_depth": [5],
                                   "learning_rate": [0.1],
                                   "n_estimators": [200],}
        else:
            self.parameters = parameters
    
    def fit(self,x,y):
        scoring = {"roc": make_scorer(roc_auc_score),}
        
        x = x.values
        y = y.values.reshape(len(y),)
        
        if self.method == 'logistic':
            clf = GridSearchCV(linear_model.LogisticRegression(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='roc')
            clf.fit(x,y)
            
            coef = pd.DataFrame(clf.best_estimator_.coef_).sort_values()
            self.select_col = coef.index[-self.TopN:]
            
        elif self.method == 'rf':
            clf = GridSearchCV(esb.RandomForestClassifier(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='roc')
            clf.fit(x,y)
            
            importances = clf.best_estimator_.feature_importances_
            Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances']).sort_values('importances')
            self.select_col = list(Rank.index[-self.Top_N:])
        
        elif self.method == 'adaBoost':
            clf = GridSearchCV(esb.AdaBoostClassifier(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='roc')
            clf.fit(x,y)
            
            importances = clf.best_estimator_.feature_importances_
            Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances']).sort_values('importances')
            self.select_col = list(Rank.index[-self.Top_N:])
            
        elif self.method == 'gbm':
            clf = GridSearchCV(esb.GradientBoostingClassifier(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='roc')
            clf.fit(x,y)
            
            importances = clf.best_estimator_.feature_importances_
            Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances']).sort_values('importances')
            self.select_col = list(Rank.index[-self.Top_N:])
            
        elif self.method == 'xgb':
            clf = GridSearchCV(XGBRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='roc')
            clf.fit(x,y)
            
            importances = clf.best_estimator_.feature_importances_
            Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances']).sort_values('importances')
            self.select_col = list(Rank.index[-self.Top_N:])
            
    def transform(self,x):
        transform_x = x.loc[:,self.choose_col].iloc[:-1,-1]
        return transform_x
        
class Feature_Mining():
    def __init__(self,method = 'poly',TopN = 10):
        self.method = method
        self.TopN = TopN
        #相关性筛选
        self.Filter_Selection = None
        #数据变换
        self.DChange = None
        #数据标准化
        self.Standard = None
        
    def fit(self,x,y):
        if self.method == 'poly':
            FS = Filter_Selection('pearson',TopN = self.TopN)
            FS.fit(x,y)
            new_x = FS.transform(x)
            Dchange = Data_Preprocess.Data_Change('poly')
            Dchange.fit(new_x)
            DChange_new_x = Dchange.transform(new_x)
            standard = Data_Preprocess.Data_Change('avgstd')
            standard.fit(DChange_new_x)

            #赋值
            self.Filter_Selection = FS
            self.DChange = Dchange
            self.Standard = standard
    
    def transform(self,x):
        FS_x = self.Filter_Selection.transform(x)
        DChange_FS_x = self.DChange.transform(FS_x)
        res = self.Standard.transform(DChange_FS_x)
        res = pd.DataFrame(res.iloc[:,self.TopN:].values,index = x.index,columns = res.columns[self.TopN:])
        res = pd.concat([x,res],axis = 1)
        return res
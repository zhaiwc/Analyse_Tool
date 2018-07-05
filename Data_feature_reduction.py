# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:50:29 2018

@author: zhaiweichen
"""
import pandas as pd
import numpy as np
import pdb

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.decomposition import PCA,FactorAnalysis,KernelPCA
from sklearn.model_selection import  GridSearchCV,cross_val_score
from sklearn.metrics import make_scorer,mean_squared_error,roc_auc_score, accuracy_score
from sklearn import linear_model
from xgboost import XGBRegressor,XGBClassifier
from Analysis_Tool import Data_plot

def dim_reduction(data,method = 'pca',n_comp = None):
    if method =='pca':
        if n_comp is None:
            pca = PCA(svd_solver= 'full',n_components = 'mle')
            pca.fit(data)
            res = pca.transform(data)
        else:
            pca = PCA(n_components=n_comp)
            pca.fit(data)
            res = pca.transform(data)
            dr = pca
            
    elif method == 'kpca':
        if n_comp is None:
            n_cmp_list = [2,3,4]  + list(range(5,data.shape[1],5))
            parameters = {
                            "n_components": n_cmp_list,
                            "kernel": ['rbf', 'poly'],
                        }
            
            scoring = {
                        "cross_val": make_scorer(cross_val_score),
                        }
            kpca = GridSearchCV(KernelPCA(),param_grid=parameters,cv=5,scoring=scoring,refit ='cross_val')
            res = kpca.fit_transform(data)
            dr = kpca
            
        else:
            kpca = KernelPCA(n_components = n_comp,kernel='rbf',gama = 10)
            res = kpca.fit_transform(data)
            dr = kpca
            
    elif method =='fa':
        if n_comp is None:
            n_cmp_list = [2,3,4]  + list(range(5,data.shape[1],5))
            parameters = {
                            "n_components": n_cmp_list,
                        }
            
            scoring = {
                        "cross_val": make_scorer(cross_val_score),
                        }
            fa = GridSearchCV(FactorAnalysis(),param_grid=parameters,cv=5,scoring=scoring,refit ='cross_val')
            res = fa.fit_transform(data)
            dr = fa
            
        else:
            fa = FactorAnalysis(n_components = n_comp)
            res = fa.fit_transform(data)
            dr = fa
            
    return res,dr

def reg_feature_selection(x,y,method ='rf',Top_N = 20,**kw):
    '''
    回归特征选择
    1.lasso
    2.随机森林特征
    3.GBM
    4.XGBOOST
    '''
    print('开始对回归因子进行特征选择,使用的方法：{}...'.format(method))
    if method == 'lasso':
        if 'alpha' in kw:
            alpha = kw['alpha']
            reg = linear_model.Lasso(alpha = alpha)
            reg.fit(x,y)
            
            coef = pd.DataFrame(reg.coef_)
            select_col = coef[coef!=0].dropna()
        else:
            parameters = {
                "alpha": [0.01,0.1,1,10,100],
            }
            scoring = {
            "mse": make_scorer(mean_squared_error),
            }
            reg = GridSearchCV(linear_model.Lasso(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
            reg.fit(x,y)
            
            coef = pd.DataFrame(reg.best_estimator_.coef_)
            select_col = coef[coef!=0].dropna()
            
    elif method =='rf':
        parameters = {
                        "max_depth": [3, 5, 7],
                        "n_estimators": [50, 100, 200],
                    }
        scoring = {
            "mse": make_scorer(mean_squared_error),
            }
#        pdb.set_trace()
        reg = GridSearchCV(RandomForestRegressor(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),))
        
        importances = reg.best_estimator_.feature_importances_
        Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances'])
        Rank = Rank.sort_values('importances')
        select_col = list(Rank.iloc[-Top_N:,:].index)
    
    elif method == 'gbm':
        parameters = {
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.01, 0.1],
                        "n_estimators": [50, 100, 200],
                    }
        scoring = {
            "mse": make_scorer(mean_squared_error),
            }
        reg = GridSearchCV(GradientBoostingRegressor(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),))
    
        importances = reg.best_estimator_.feature_importances_
        Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances'])
        Rank = Rank.sort_values('importances')
        select_col = list(Rank.iloc[-Top_N:,:].index)
        
    elif method =='xgb':
        parameters = {
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.01, 0.1],
                        "n_estimators": [50, 100, 200],
                    }
        scoring = {
            "mse": make_scorer(mean_squared_error),
            }
        reg = GridSearchCV(XGBRegressor(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),))
#        pdb.set_trace()
#        importances = pd.Series(reg.best_estimator_.booster().get_fscore()).sort_values(ascending =False)
        importances = pd.Series(reg.best_estimator_.booster().get_score(importance_type='weight')).sort_values(ascending =False)
#        indices = np.argsort(importances)[::-1]
        select_col = list(importances.index)
    return select_col

def clf_feature_selection(x,y,method ='rf',Top_N = 20,**kw):
    '''
    回归特征选择
    1.随机森林特征
    2.GBM
    3.XGBOOST
    '''
    print('开始对分类因子进行特征选择,使用的方法：{}...'.format(method))
    if method =='rf':
        parameters = {
                        "max_depth": [3, 5, 7],
                        "n_estimators": [50, 100, 200],
                    }
        scoring = {
            "AUC": make_scorer(roc_auc_score),
            "Accuracy": make_scorer(accuracy_score)
            }
        clf = GridSearchCV(RandomForestClassifier(),param_grid=parameters,cv=5,scoring=scoring,refit ='AUC')
        clf.fit(np.array(x),np.array(y).reshape(len(y),))
        
        importances = clf.best_estimator_.feature_importances_
        Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances'])
        Rank = Rank.sort_values('importances')
        select_col = list(Rank.iloc[-Top_N:,:].index)
        plt = Data_plot.plot_barh(Rank.iloc[-Top_N:,:])
        plt.title('Feature importances')
        plt.ylabel('Feature')
        plt.xlabel('importances')
    
    elif method == 'gbm':
        parameters = {
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.01, 0.1],
                        "n_estimators": [50, 100, 200],
                    }
        scoring = {
            "AUC": make_scorer(roc_auc_score),
            "Accuracy": make_scorer(accuracy_score)
            }
        clf = GridSearchCV(GradientBoostingClassifier(),param_grid=parameters,cv=5,scoring=scoring,refit ='AUC')
        clf.fit(np.array(x),np.array(y).reshape(len(y),))
    
        importances = clf.best_estimator_.feature_importances_
        Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances'])
        Rank = Rank.sort_values('importances')
        select_col = list(Rank.iloc[-Top_N:,:].index)
        
    elif method =='xgb':
        parameters = {
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.01, 0.1],
                        "n_estimators": [50, 100, 200],
                    }
        scoring = {
            "AUC": make_scorer(roc_auc_score),
            "Accuracy": make_scorer(accuracy_score)
            }
        clf = GridSearchCV(XGBClassifier(),param_grid=parameters,cv=5,scoring=scoring,refit ='AUC')
        clf.fit(np.array(x),np.array(y).reshape(len(y),))
        
        importances = clf.best_estimator_.feature_importances_
        Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances'])
        Rank = Rank.sort_values('importances')
        select_col = list(Rank.iloc[-Top_N:,:].index)
        
    return select_col
        
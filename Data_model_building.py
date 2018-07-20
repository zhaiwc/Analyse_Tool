# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 11:21:11 2018

@author: zhaiweichen
"""
from Analysis_Tool import Data_plot,Data_Preprocess,Data_analysis,Data_feature_reduction
from sklearn import linear_model,kernel_ridge,tree,svm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  GridSearchCV,train_test_split
from sklearn.metrics import make_scorer,mean_squared_error,r2_score,roc_auc_score,accuracy_score,v_measure_score
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor,XGBClassifier
from collections import Counter

import pdb
import pandas as pd
import numpy as np

def split_train_test(x,y,test_size = 0.2,random_state = None):
    '''
    拆分数据训练集，测试集
    '''
    X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def cls_sample_balance(x,y,method ='random',rank_by = None,Multiple = 4,boostrap =False,benchmark = 'min'):
    '''
    样本均衡：
    通过筛选，使得正负样本尽量均衡。保证各类样本比例不超过4:1,by label 。
    method = 'near':如果样本失衡，则根据排序类别的顺序下，取最近的样本。
    method = 'random':如果样本失衡，则在多数类别中，进行随机抽取。
    Multiple:大类样本与小类样本数量比
    boostrap：是否重采样
    benchmark：max :以大类为基准 ，min : 以小类为基准
    '''
    print('-----开始进行样本均衡-----')
    #统计y的count()
    cnt = Counter(y.iloc[:,0])
    print('当前分类信息：',cnt)
    x_res = pd.DataFrame()
    y_res = pd.DataFrame()
    if benchmark == 'min':
        #计算最小的类和类的个数
        min_num = min(cnt.values())
        for key in cnt.keys():
            if cnt[key] == min_num:
                minclass = key
    样本
        for key in cnt.keys():
            if cnt[key] < min_num * Multiple:
                idx = y[y==key].dropna().index
            else:
                if method =='near':
                    #合并x,y，并排序
                    data = pd.concat([x,y],axis =1)
                    data = data.sort_values(rank_by)
                    #计算最少类的idx,获取最小类rank列的范围
                    idx_short = y[y==minclass].dropna().index
                    max_idx = data.loc[idx_short[0] ,rank_by]
                    min_idx = data.loc[idx_short[-1],rank_by]
                    #计算当前过多的类的idx
                    idx_long = y[y==key].dropna().index
                    data = data.loc[idx_long,:]
                    if len(data[(data[rank_by]>min_idx) & (data[rank_by]<max_idx)]) > min_num * Multiple:
                        idx = data[data[rank_by]>min_idx & data[rank_by]<max_idx].sample(int(min_num * Multiple),replace=boostrap).index
                    else:
                        #如果最少类所夹的范围不足以取相应的数据，则在数据前后补齐。
                        idx = list(data[(data[rank_by]>min_idx) & (data[rank_by]<max_idx)].index)
                        part2len = abs(int((len(data[(data[rank_by]>min_idx) & (data[rank_by]<max_idx)]) - min_num * Multiple) /2)) 
                        if part2len > len(data[data[rank_by]<min_idx]):
                            idx = idx + list(data[data[rank_by]<min_idx].index)                   
                        else:
                            idx = idx + list(data[data[rank_by]<min_idx].sample(int(part2len),replace=boostrap).index)
                        if part2len > len(data[data[rank_by]>max_idx]):
                            idx = idx + list(data[data[rank_by]>max_idx].index)
                        else:
                            idx = idx + list(data[data[rank_by]>max_idx].sample(int(part2len),replace=boostrap).index)
                    
                elif method == 'random':
                    idx = y[y==key].dropna().sample(int(min_num * Multiple),replace=boostrap).index
                    
            x_new = x.loc[idx,:]
            y_new = y.loc[idx,:]
            x_res = pd.concat([x_res,x_new])
            y_res = pd.concat([y_res,y_new])
            
            #统计结果y的count()
            cnt_res = Counter(y_res.iloc[:,0])
            print('样本均衡后分类信息：',cnt_res)
    
    elif benchmark == 'max': 
        #计算最大的类和类的个数
        max_num = max(cnt.values())
        for key in cnt.keys():
            if cnt[key] > max_num / Multiple:
                idx = y[y==key].dropna().index
            else:
                #从小数目中采集多个样本只能重采样。                  
                idx = y[y==key].dropna().sample(int(max_num / Multiple),replace=True).index
                    
            x_new = x.loc[idx,:]
            y_new = y.loc[idx,:]
            x_res = pd.concat([x_res,x_new])
            y_res = pd.concat([y_res,y_new])
            
            #统计结果y的count()
            cnt_res = Counter(y_res.iloc[:,0])
            print('样本均衡后分类信息：',cnt_res)
    
    return x_res,y_res

def reg_model(x,y,method = 'linear',parameters = None):
#    print('-----开始构建回归模型，方法：{}-----'.format(method))
    if method  == 'linear':
        reg = linear_model.LinearRegression()     
        reg.fit(x,y)
        parameters = None
        
    elif method == 'ridge':
        parameters = {
                "alpha": [0.01,0.1,1,10,100],
                    }
        scoring = {
                "mse": make_scorer(mean_squared_error),
                }
        reg = GridSearchCV(linear_model.Ridge(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),))
        
    elif method == 'lasso':
        parameters = {
                "alpha": [0.01,0.1,1,10,100],
                    }
        scoring = {
                "mse": make_scorer(mean_squared_error),
                }
        reg = GridSearchCV(linear_model.Lasso(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),))
        parameters = reg.best_params_
        
    elif  method == 'elasticnet':
        parameters = {
                "alpha": [0.1,1,10],
                "l1_ratio":[.1, .5,.9]
                    }
        scoring = {
                "mse": make_scorer(mean_squared_error),
                }
        reg = GridSearchCV(linear_model.ElasticNet(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),))
        parameters = reg.best_params_
        
    elif method == 'kernelridge':
        parameters = {
                "alpha": [0.01,0.1,1,10,100],
                "gamma":[3,5,7]
                    }
        scoring = {
                "mse": make_scorer(mean_squared_error),
                }

        reg = GridSearchCV(kernel_ridge.KernelRidge(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),))
        
        parameters = reg.best_params_
    
    elif method == 'svr':
        if parameters is not None:
            for key in parameters.keys():
                if type(parameters[key]) is list:
                    pass
                else:
                    parameters[key] = [parameters[key]]
        else:
            parameters = {
                            "C": [10,100,1000,10000],
                            "epsilon": [10,1,0.1,0.01],

                        }
            scoring = {
                "mse": make_scorer(mean_squared_error),
                }
        reg = GridSearchCV(SVR(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),))

        parameters = reg.best_params_
        
    elif method == 'rf':
        if parameters is not None:
            for key in parameters.keys():
                if type(parameters[key]) is list:
                    pass
                else:
                    parameters[key] = [parameters[key]]
        else:
            parameters = {
                            "max_depth": [3, 5, 7],
                            "n_estimators": [300,500,1000],
                        }
        scoring = {
                "mse": make_scorer(mean_squared_error),
                }
        reg = GridSearchCV(RandomForestRegressor(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),))
        
        parameters = reg.best_params_
    
    elif method == 'gbm':
        if parameters is not None:
            for key in parameters.keys():
                if type(parameters[key]) is list:
                    pass
                else:
                    parameters[key] = [parameters[key]]
        else:
            parameters = {
                            "max_depth": [3, 5],
                            #"learning_rate": [0.01, 0.1],
                            "n_estimators": [500,1000],
                        }
        scoring = {
                "mse": make_scorer(mean_squared_error),
                }
        reg = GridSearchCV(GradientBoostingRegressor(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),))
        
        parameters = reg.best_params_
    
    elif method =='xgb':
        if parameters is not None:
            for key in parameters.keys():
                if type(parameters[key]) is list:
                    pass
                else:
                    parameters[key] = [parameters[key]]
        else:
            parameters = {
                            "max_depth": [3, 5],
                            "learning_rate": [0.01, 0.1],
                            "n_estimators": [500,1000],
                        }
        scoring = {
                "mse": make_scorer(mean_squared_error),
                }
        reg = GridSearchCV(XGBRegressor(),param_grid=parameters,cv=5,scoring=scoring,refit ='mse')
        reg.fit(np.array(x),np.array(y).reshape(len(y),).astype(float))
        
        parameters = reg.best_params_
    return reg,parameters

def cls_model(x,y,method = 'logistic',parameters = None,Top_N = 20):
    if method  == 'logistic':
        cls = linear_model.LogisticRegression()
        cls.fit(np.array(x),np.array(y).reshape(len(y),))
        
    elif method  == 'knn':
        if parameters is not None:
            for key in parameters.keys():
                if type(parameters[key]) is list:
                    pass
                else:
                    parameters[key] = [parameters[key]]
        else:
            parameters = {
                    "n_neighbors": [3,4,5,6,7],
                    "weights":['uniform','distance'],
                    'algorithm':['auto','ball_tree','kd_tree','brute']
                        }
        scoring = {
                "AUC": make_scorer(roc_auc_score),
                "Accuracy": make_scorer(accuracy_score)
                }
        cls = GridSearchCV(KNeighborsClassifier(),param_grid=parameters,cv=5,scoring=scoring,refit ='Accuracy')
        cls.fit(np.array(x),np.array(y).reshape(len(y),))
        
    elif method == 'dt':
        if parameters is not None:
            for key in parameters.keys():
                if type(parameters[key]) is list:
                    pass
                else:
                    parameters[key] = [parameters[key]]
        else:
            parameters = {
                    "criterion": ['gini','entropy'],
                    "max_depth":[2,3,4,5],
                    'max_features':['auto','log2',None]
                        }
        scoring = {
                "AUC": make_scorer(roc_auc_score),
                "Accuracy": make_scorer(accuracy_score)
                }
        cls = GridSearchCV(tree.DecisionTreeClassifier(),param_grid=parameters,cv=5,scoring=scoring,refit ='Accuracy')
        cls.fit(np.array(x),np.array(y).reshape(len(y),))
        
    elif method == 'svm':
        if parameters is not None:
            for key in parameters.keys():
                if type(parameters[key]) is list:
                    pass
                else:
                    parameters[key] = [parameters[key]]
        else:
            parameters = {
                    "C" : [0.1,1,10],
                    "kernel" : ['linear','rbf','sigmoid'],
                    "degree" : [2,3,4],
                        }
        scoring = {
                "AUC": make_scorer(roc_auc_score),
                "Accuracy": make_scorer(accuracy_score)
                }
        cls = GridSearchCV(svm.SVC(),param_grid=parameters,cv=5,scoring=scoring,refit ='Accuracy')
        cls.fit(np.array(x),np.array(y).reshape(len(y),))
        
    elif method == 'rf':
        if parameters is not None:
            for key in parameters.keys():
                if type(parameters[key]) is list:
                    pass
                else:
                    parameters[key] = [parameters[key]]
        else:
            parameters = {
                            "max_depth": [3, 5, 7],
                            "n_estimators": [300,500,1000],
                        }
        scoring = {
            "AUC": make_scorer(roc_auc_score),
            "Accuracy": make_scorer(accuracy_score)
            }
        cls = GridSearchCV(RandomForestClassifier(),param_grid=parameters,cv=10,scoring=scoring,refit ='AUC')
        cls.fit(np.array(x),np.array(y).reshape(len(y),))
        
        importances = cls.best_estimator_.feature_importances_
        Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances'])
        Rank = Rank.sort_values('importances')
        key_feature = Rank.iloc[-Top_N:,:]
        
    elif method == 'gbm':
        if parameters is not None:
            for key in parameters.keys():
                if type(parameters[key]) is list:
                    pass
                else:
                    parameters[key] = [parameters[key]]
        else:
            parameters = {
                            "max_depth": [3, 5],
                            "learning_rate": [0.01, 0.1],
                            "n_estimators": [500,1000],
                        }
        scoring = {
            "AUC": make_scorer(roc_auc_score),
            "Accuracy": make_scorer(accuracy_score)
            }
        cls = GridSearchCV(GradientBoostingClassifier(),param_grid=parameters,cv=10,scoring=scoring,refit ='AUC')
        cls.fit(np.array(x),np.array(y).reshape(len(y),))
        importances = cls.best_estimator_.feature_importances_
        Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances'])
        Rank = Rank.sort_values('importances')
        key_feature = Rank.iloc[-Top_N:,:]
        
    elif method =='xgb':
        if parameters is not None:
            for key in parameters.keys():
                if type(parameters[key]) is list:
                    pass
                else:
                    parameters[key] = [parameters[key]]
        else:
            parameters = {
                            "max_depth": [3, 5, 7],
                            "learning_rate": [0.01, 0.1],
                            "n_estimators": [500,1000],
                        }
        scoring = {
            "AUC": make_scorer(roc_auc_score),
            "Accuracy": make_scorer(accuracy_score)
            }
        cls = GridSearchCV(XGBClassifier(),param_grid=parameters,cv=10,scoring=scoring,refit ='AUC')
        cls.fit(np.array(x),np.array(y).ravel())
        
        importances = cls.best_estimator_.feature_importances_
        Rank = pd.DataFrame(importances,index=x.columns,columns = ['importances'])
        Rank = Rank.sort_values('importances')
        key_feature = Rank.iloc[-Top_N:,:]
        
    return cls,cls.best_params_,key_feature

def clu_model(x,method = 'kmeans',parameters = None):
    if method == 'kmeans':
        clu = KMeans(parameters)
        clu.fit(x)
        
    elif method == 'DBSCAN':#密度聚类
        clu = DBSCAN(parameters)
        clu.fit(x)
        
    elif method =='Agg':#凝聚聚类
        clu = AgglomerativeClustering(parameters)
        clu.fit(x)
        
    return clu

#class y_change():
#    '''
#    y值变换
#    '''
#    def __init__(self):
#        self.method = None
#        self.std = None
#        self.mean = None
#        self.max = None
#        self.min = None
#        
#    def transform_y(self,y,method = 'log'):
#        if method == 'log':
#            self.method = 'log'
#            y_new = Data_Preprocess.data_change(y,method = method)
#        elif method == 'avgstd':
#            self.method = 'avgstd'
#            y_new,scaler = Data_Preprocess.data2avgstd(y)
#            self.mean = scaler.mean_
#            self.std = scaler.scale_            
#        elif method == 'minmax':
#            self.method = 'minmax'
#            y_new,scaler = Data_Preprocess.data2avgstd(y)
#            self.max = scaler.data_max_
#            self.min = scaler.data_min_
#        return y_new
#
#    def change_back_y(self,y):
#        '''
#        如对y进行转换，则需要将y转换回来
#        '''
#        if self.method == 'log':
#            y = np.exp(y)
#        elif self.method == 'avgstd':
#            y = y * self.std + self.mean
#        elif self.method == 'minmax':
#            y = y *(self.max - self.min)  + self.min
#        
#        return y

def reg_score(reg,train_x,train_y,valid_x,valid_y,label = None,is_plot = True,
              y_change = None,**kw):
    '''
    对回归模型进行评价
    不分label: 对所有数据进行拟合预测，画散点折线图，计算mse,r2指标
    输入：
    reg:回归模型
    train_x,train_y,valid_x,valid_y：训练，验证 的x,y
    is_plot:是否输入图表
        
    '''
    if label is not None:
        train_x_input = train_x.drop(label,axis = 1)
        valid_x_input = valid_x.drop(label,axis = 1)
    else:
        train_x_input = train_x
        valid_x_input = valid_x
    if y_change is None:
        train_pred_y = reg.predict(train_x_input)
    else:
        train_pred_y = y_change.change_back(pd.DataFrame(reg.predict(train_x_input),columns=['train_pred_y'],index = train_y.index))
        train_y = y_change.change_back(train_y)
    train_mse = mean_squared_error(train_y,train_pred_y)
    train_r2 = r2_score(train_y,train_pred_y)
    train_pred_y = pd.DataFrame(train_pred_y,columns=['train_pred_y'],index = train_y.index)

    #画y预测与y真实 按原顺序比较
    if is_plot:
        plt = Data_plot.plot_scatter(train_y)   
        plt = Data_plot.plot_line(train_pred_y,c=['r--'])
        plt.show()
    
    
        plot_train_data = pd.concat([train_y,train_pred_y],axis=1)
        plt = Data_plot.plot_scatter(plot_train_data) 
        line_data = np.array([[plot_train_data.max()[0],plot_train_data.max()[0]],[plot_train_data.min()[0],plot_train_data.min()[0]]])
        plt = Data_plot.plot_line(pd.DataFrame(line_data,columns=['y_true','y_pred']))
        plt.show()

    print('训练集：mse = {} , r2 = {}'.format(train_mse,train_r2))
    
    if y_change is None:
        valid_pred_y = reg.predict(valid_x_input)
    else:
        valid_pred_y = y_change.change_back(pd.DataFrame(reg.predict(valid_x_input),columns=['valid_pred_y'],index = valid_y.index))
        valid_y = y_change.change_back(valid_y)
    valid_mse = mean_squared_error(valid_y,valid_pred_y)
    valid_r2 = r2_score(valid_y,valid_pred_y)
    valid_pred_y = pd.DataFrame(valid_pred_y,columns=['valid_pred_y'],index = valid_y.index)
    
    if is_plot:
        plt = Data_plot.plot_scatter(valid_y)
        plt = Data_plot.plot_line(valid_pred_y,c=['r--'])
        plt.show()
        
        
        plot_valid_data = pd.concat([valid_y,valid_pred_y],axis=1)
        plt = Data_plot.plot_scatter(plot_valid_data) 
        line_data = np.array([[plot_valid_data.max()[0],plot_valid_data.max()[0]],[plot_valid_data.min()[0],plot_valid_data.min()[0]]])
        plt = Data_plot.plot_line(pd.DataFrame(line_data,columns=['y_true','y_pred']),)
        plt.show()
    print('验证集：mse = {} , r2 = {}'.format(valid_mse,valid_r2))
    return valid_mse,valid_r2
    
def cls_scors(cls,train_x,train_y,valid_x,valid_y,label = None):
    '''
    对分类模型进行评价
    '''
    if label is not None:
        train_x_input = train_x.drop(label,axis = 1)
        valid_x_input = valid_x.drop(label,axis = 1)
    else:
        train_x_input = train_x
        valid_x_input = valid_x
        
    train_pred_y = cls.predict(train_x_input)
    train_acc = accuracy_score(train_y,train_pred_y)
    train_auc = roc_auc_score(train_y,train_pred_y)
    train_pred_y = pd.DataFrame(train_pred_y,columns=['train_pred_y'],index = train_y.index)
    
    plt = Data_plot.plot_confusion_matrix(train_y,train_pred_y)   
    plt.show()
    print('训练集：acc = {} , auc = {}'.format(train_acc,train_auc))
    
    valid_pred_y = cls.predict(valid_x_input)
    valid_acc = accuracy_score(valid_y,valid_pred_y)
    valid_auc = roc_auc_score(valid_y,valid_pred_y)
    valid_pred_y = pd.DataFrame(valid_pred_y,columns=['valid_pred_y'],index = valid_y.index)
    
    plt = Data_plot.plot_confusion_matrix(valid_y,valid_pred_y)   
    plt.show()
    print('验证集：acc = {} , auc = {}'.format(valid_acc,valid_auc))

class pca_clu_reg():
    '''
    实现降维，聚类，再回归的方式
    '''
    def __init__(self,para):
        #进入模型的维度
        self.dim = para['dim'] 
        #降维的模型，PCA
        self.dr_model = None
        #聚类类数
        self.n_clu = para['n_clu']
        self.clu_model = None
        self.method = para['method']
        self.reg_model = {}
        self.reg_para = {}    
        
    def fit(self,x,y):
        #降维
        x_dr,dr = Data_feature_reduction.dim_reduction(x,n_comp = self.dim)
        #聚类
#        pdb.set_trace()
        clu = KMeans(n_clusters=self.n_clu).fit(x_dr[:,[0,1]])
        clu_label = clu.predict(np.array(x_dr[:,[0,1]]))     
        clu_label = pd.DataFrame(clu_label,columns = ['label'],index = x.index)

        self.dr_model = dr
        self.clu_model = clu
        
        for i in range(self.n_clu):
            x_dr,dr = Data_feature_reduction.dim_reduction(x,n_comp = self.dim)
            x_dr = pd.DataFrame(x_dr,index = x.index)
            clu_data = pd.concat([x_dr,clu_label],axis = 1)
            
            x_clu = clu_data[clu_data.label ==i]
            y_clu = y.loc[x_clu.index,:]
            #创建模型
            reg,para = reg_model(x_dr.loc[x_clu.index,:],y_clu,method= self.method)
            self.reg_model[i] = reg
            self.reg_para[i] = para
            
    def predict(self,x):
        dr = self.dr_model
        clu = self.clu_model
        #降维
        x_dr = dr.transform(x)
        x_dr = pd.DataFrame(x_dr)
        #聚类
        clu_label = clu.predict(x_dr.iloc[:,[0,1]])
        clu_label = pd.DataFrame(clu_label,columns = ['label'])
        
        clu_data = pd.concat([x_dr,clu_label],axis = 1)
        res = []
        for i in range(self.n_clu):
            x_clu = clu_data[clu_data.label ==i]
            if len(x_clu):
                reg_model = self.reg_model[i]
                y_pred = reg_model.predict(np.array(x_dr.loc[x_clu.index,:]))
                y_pred = pd.DataFrame(y_pred,index = x_clu.index)
                res.append(y_pred)
        res = pd.concat(res).sort_index()
        res = res.rename(columns = {res.columns[0]:'y_pred'})
        return np.array(res)
            
            
class reg_stack():    
    '''
    对相同的模型进行stack
    '''
    def __init__(self,n_model = 10,min_mse = None,pca_clu = False,para ={'method':'linear'},y_change = None):
        self.n_model = n_model
        self.model_list = []
        self.min_mse = min_mse
        self.pca_clu = pca_clu
        self.para = para
        self.mse_list = []
        self.y_change = y_change

    def fit(self,train_valid_x,train_valid_y):
        for i in range(self.n_model):
            #拆分训练集，验证集
            train_x,valid_x,train_y,valid_y = Data_Preprocess.split_train_test(train_valid_x,train_valid_y)
            if self.pca_clu:
                reg = pca_clu_reg(self.para)
                reg.fit(train_x,train_y)
                
                if self.y_change is None:
                    mse = mean_squared_error(valid_y,reg.predict(valid_x))
                else:
                    mse = mean_squared_error(self.y_change.change_back(valid_y),self.y_change.change_back(pd.DataFrame(reg.predict(valid_x))))

                if self.min_mse is None:
                    self.model_list.append(reg)
                    self.mse_list.append(mse)
                else:
                    if mse < self.min_mse:
                        self.model_list.append(reg)
                        self.mse_list.append(mse)
            else:#普通方法
                pass
                        
    def predict(self,x):
        res = []
        for model in self.model_list:
            res.append(pd.DataFrame(model.predict(x)))
        res = np.array(pd.concat(res,axis=1).mean(axis=1))
        return res

class cls_model_stack():
    '''
    cls_list : 堆叠模型的类型
    n_cls: 堆叠模型的个数
    cls_ratio：各个模型所占比例
    '''
    def __init__(self,cls_list,Multiple = 4,n_cls = 10,cls_ratio = None ,boostrap = False,benchmark = 'min'):
        self.cls_list = cls_list
        self.Multiple = Multiple
        self.n_cls = n_cls
        self.modellist = []
        self.paralist = []
        self.keyfeatures = {}
        self.benchmark = benchmark
        self.boostrap = boostrap
        if cls_ratio is None:
        #默认平均分配
            self.cls_ratio = [1 / len(cls_list)] * len(cls_list)
        else:
            self.cls_ratio = cls_ratio
            
    def fit(self,train_x,train_y):
        '''
        拟合数据,计算单个模型关键因子
        '''
        print('-----开始进行模型stack训练,模型个数：{}-----'.format(self.n_cls))
        
        for i,cls in enumerate(self.cls_list):
            best_para = None
            for echo in range(int(self.n_cls * self.cls_ratio[i])):
                #进行样本均衡，有放回随机抽样
                betch_x,betch_y = Data_Preprocess.sample_balance(train_x,train_y,
                                  Multiple = self.Multiple,boostrap = self.boostrap,benchmark = self.benchmark)
                #对单一算法进行拟合
                cls_m,para,keyfeature = cls_model(betch_x,betch_y,method = cls ,parameters = best_para)
                best_para = para
                self.keyfeatures[str(cls) + str(echo)] = keyfeature
                print(best_para)
            self.modellist.append(cls_m)
            self.paralist.append(para)
                
    def predict(self,x):
        '''
        预测数据,根据输入的数据进行预测，最后通过结果投票得到最终结果。
        '''
        print('-----开始对stack模型进行数据预测-----')
        
        res = pd.DataFrame()
        for i,cls in enumerate(self.cls_list):
            cls = self.modellist[i]
            for echo in range(int(self.n_cls * self.cls_ratio[i])):
                ths_res = pd.DataFrame(cls.predict(x))
                res = pd.concat([res,ths_res],axis =1 )
        #使用投票的方式进行，获取类别
        score_df = pd.DataFrame()
        label = set(res.iloc[:,0])
        for lab in label:
            #计算每个样本
            score = pd.DataFrame(res[res == lab].count(axis=1)/len(label),columns = [lab]) 
            score_df = pd.concat([score_df,score],axis = 1)
        #根据得分，投票计算最终结果
        res_list = []     
        for i in range(len(score_df)):
            res_list.append(score_df.iloc[i,:][score_df.iloc[i,:] == score_df.iloc[i,:].max()].index[0])
        res_fanl = np.array(res_list)

        return res_fanl
    
    def static_keyfeature(self,method = 'sum',TopN = 20):
        '''
        统计关键因子
        '''
        print('-----开始对stack模型进行关键因子统计-----')
        if method == 'count':
            keyfea_cnt = Counter()
            if len(self.keyfeatures):
                for key in self.keyfeatures.keys():
                    for feat in self.keyfeatures[key].index:
                        if feat in keyfea_cnt:
                            keyfea_cnt[feat] += 1
                        else:
                            keyfea_cnt[feat] = 1
                keyfea_cnt = keyfea_cnt.most_common(TopN) 
                #根据关键因子出现次数，画图
                keyfea_df = pd.DataFrame(keyfea_cnt).set_index(0)
            else:
                keyfea_df = None
                
        elif method == 'sum':
            keyfea_df = pd.DataFrame()
            if len(self.keyfeatures):
                for key in self.keyfeatures.keys():
                    keyfea_df = pd.concat([keyfea_df,self.keyfeatures[key]],axis=1)
                keyfea_df = pd.DataFrame(keyfea_df.sum(axis=1).sort_values(),columns=['importances'])
                keyfea_df = keyfea_df.iloc[0:TopN,:]
            else:
                keyfea_df = None
        
        plt = Data_plot.plot_barh(keyfea_df)
        plt.title('Feature importances')
        plt.ylabel('Feature')
        plt.xlabel('importances')

        return keyfea_df
    

                
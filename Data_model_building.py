# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 11:21:11 2018

@author: zhaiweichen
"""
from Analysis_Tool import Data_plot,Data_Preprocess,Data_feature_reduction
from sklearn import linear_model,tree,svm,neural_network 
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.model_selection import  GridSearchCV,train_test_split
from sklearn.metrics import make_scorer,mean_squared_error,r2_score,roc_auc_score,accuracy_score
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from xgboost import XGBRegressor,XGBClassifier
from collections import Counter,defaultdict
from mlxtend.regressor import StackingRegressor
from mlxtend.classifier import StackingClassifier
from sklearn.preprocessing import label_binarize
import sklearn.ensemble as esb
import pdb
import copy
import time
import pandas as pd
import numpy as np


def split_train_test(x,y,test_size = 0.2,random_state = None):
    '''
    拆分数据训练集，测试集
    '''
    X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def reg_sample_balance(x,y,benchmark = 'min',Multiple = 3,boostrap = False):
    '''
    回归样本均衡
    '''
    #划分上下极端样本，连续样本离散化
    label_y = copy.copy(y)
    label_y[ label_y > label_y.mean() + label_y.std()] = 1
    label_y[ label_y < label_y.mean() - label_y.std()] = 3
    label_y[ (label_y <= label_y.mean() - label_y.std()) & (label_y >= label_y.mean() - label_y.std())] = 2
    #统计y的count()
    cnt = Counter(label_y.iloc[:,0])
    #print('当前分类信息：',cnt)
    x_res = pd.DataFrame()
    y_res = pd.DataFrame()
    if benchmark == 'min':
        #计算最小的类和类的个数(上采样)
        min_num = min(cnt.values())

        for key in cnt.keys():
            if cnt[key] < min_num * Multiple:
                idx = label_y[label_y==key].dropna().index
            else:
                idx = label_y[label_y==key].dropna().sample(int(min_num * Multiple),replace=boostrap).index
                    
            x_new = x.loc[idx,:]
            y_new = y.loc[idx,:]
            x_res = pd.concat([x_res,x_new])
            y_res = pd.concat([y_res,y_new])
            
    elif benchmark == 'max': 
        #计算最大的类和类的个数（下采样）
        max_num = max(cnt.values())
        for key in cnt.keys():
            if cnt[key] > max_num / Multiple:
                idx = label_y[label_y==key].dropna().index
            else:
                #从小数目中采集多个样本只能重采样。                  
                idx = label_y[label_y==key].dropna().sample(int(max_num / Multiple),replace=True).index
                    
            x_new = x.loc[idx,:]
            y_new = y.loc[idx,:]
            x_res = pd.concat([x_res,x_new])
            y_res = pd.concat([y_res,y_new])
            
    elif benchmark == 'all':
        #对全部样本进行采样，只能进行重采样
        idx = y.dropna().sample(len(y),replace=True).index
        
        x_res = x.loc[idx,:]
        y_res = y.loc[idx,:]
            
    return x_res,y_res

class reg_model():
    def __init__(self,method,isGridSearch = True):
        #设置基本参数
        self.method = method
        self.isGridSearch = isGridSearch
        #设置缺省参数
        self.reg_model = None
        self.parameters = None
        self.factor_name = None
        self.best_parameters = None
    
    def set_parameters(self,parameters = None):
        #设置模型参数:如果需要调参，则自定义的是多组参数调参范围，如果不需要调参，则自定义一组参数即可
        #parameters dict like
        if parameters is None: #用户不传参数，则使用默认参数
            if self.isGridSearch == True:
                if self.method == 'linear':
                    self.parameters = None
                elif self.method == 'ridge':
                    self.parameters = {'alpha':[0.01,0.1,1,10,100]}
                elif self.method == 'lasso':
                    self.parameters = {'alpha':[0.01,0.1,1,10,100]}
                elif self.method == 'ElasticNet':
                    self.parameters = {"alpha": [0.1,1,10],
                                       "l1_ratio":[.1, .5,.9]}
                elif self.method == 'pls':
                    self.parameters = {'n_components':[3,5,7]}
                elif self.method == 'svr':
                    self.parameters = {"C": [0.1,1,10,100],
                                       "epsilon": [10,1,0.1,0.01]}
                elif self.method == 'knn':
                    self.parameters = {'n_neighbors':[3,5,7]}
                elif self.method == 'dt':
                    self.parameters = {'max_depth' :[3,5,7]}
                elif self.method == 'rf':
                    self.parameters = {"max_depth": [3, 5, 7],
                                      "n_estimators": [300,500,1000],}
                elif self.method == 'adaBoost':
                    self.parameters = { "learning_rate": [0.01, 0.1],
                                      "n_estimators": [500,1000],}
                elif self.method == 'gbm':
                    self.parameters = {"max_depth": [3, 5],
                                      "learning_rate": [0.01, 0.1],
                                      "n_estimators": [500,1000],}
                elif self.method == 'xgb':
                    self.parameters = {"max_depth": [3, 5],
                                      "learning_rate": [0.01, 0.1],
                                      "n_estimators": [500,1000],}
                elif self.method == 'bp':
                    self.parameters = {'activation':['relu'],
                                       'hidden_layer_sizes' : [(10,),(20,),(100,)],
                                       'max_iter': [200000],}
            else:
                if self.method == 'linear':
                    self.parameters = None
                elif self.method == 'ridge':
                    self.parameters = {'alpha':[1]}
                elif self.method == 'lasso':
                    self.parameters = {'alpha':[1]}
                elif self.method == 'ElasticNet':
                    self.parameters = {"alpha": [1],
                                       "l1_ratio":[0.5]}
                elif self.method == 'pls':
                    self.parameters = {'n_components':[5]}
                elif self.method == 'svr':
                    self.parameters = {"C": [1],
                                       "epsilon": [0.1]}
                elif self.method == 'knn':
                    self.parameters ={'n_neighbors':[5]}
                elif self.method == 'dt':
                    self.parameters ={'max_depth' :[5]}
                elif self.method == 'rf':
                    self.parameters ={"max_depth": [5],
                                      "n_estimators": [500],}
                elif self.method == 'adaBoost':
                    self.parameters ={ "learning_rate": [0.1],
                                      "n_estimators": [500],}
                elif self.method == 'gbm':
                    self.parameters ={"max_depth": [5],
                                      "learning_rate": [0.1],
                                      "n_estimators": [500],}
                elif self.method == 'xgb':
                    self.parameters ={"max_depth": [5],
                                      "learning_rate": [0.1],
                                      "n_estimators": [500],}
                elif self.method == 'bp':
                    self.parameters = {
                                       'activation':['relu'],
                                       'hidden_layer_sizes' :[(10,),],
                                       'max_iter': [200000],}
                    
        else:#用户传参数，则以用户参数为准
            self.parameters = parameters

                
    def fit(self,x,y):
        x_train = np.array(x)
        y_train = np.array(y).reshape(y.shape[0],)
        self.factor_name = list(x.columns)
        
        if self.parameters is None:
            self.set_parameters()
        
        scoring = {"mse": make_scorer(mean_squared_error),}
        
        self.set_parameters()
        if self.method == 'linear':
            self.reg_model = linear_model.LinearRegression()     
            self.reg_model.fit(x_train,y_train)
            
        elif self.method == 'ridge':   
            self.reg_model = GridSearchCV(linear_model.Ridge(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
            
        elif self.method == 'lasso':
            self.reg_model = GridSearchCV(linear_model.Lasso(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
            
        elif self.method == 'ElasticNet':
            self.reg_model = GridSearchCV(linear_model.ElasticNet(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
        
        elif self.method == 'pls':
            self.reg_model = GridSearchCV(PLSRegression(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
        
        elif self.method == 'svr':
            self.reg_model = GridSearchCV(svm.SVR(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
            
        elif self.method == 'knn':
            self.reg_model = GridSearchCV(KNeighborsRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
        
        elif self.method == 'dt':
            self.reg_model = GridSearchCV(tree.DecisionTreeRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
            
        elif self.method == 'rf':
            self.reg_model = GridSearchCV(esb.RandomForestRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
            
        elif self.method == 'adaBoost':
            self.reg_model = GridSearchCV(esb.AdaBoostRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
            
        elif self.method == 'gbm':
            self.reg_model = GridSearchCV(esb.GradientBoostingRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
        
        elif self.method == 'xgb': 
            self.reg_model = GridSearchCV(XGBRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
        
        elif self.method =='bp':
            self.reg_model = GridSearchCV(neural_network.MLPRegressor(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='mse')
            self.reg_model.fit(x_train,y_train)
            
    def predict(self,x):
        #模型预测
        x_pred = np.array(x)
        return self.reg_model.predict(x_pred)
    
    def get_vip(self,isplot = True):
        #计算关键因子，
        if self.method in ['svr','knn','dt','bp']:
            #上述算法没有办法衡量重要因子
            return None
        else:
            col_name = 'variable importance'
            if self.method in ['linear'] :
                var_importance = pd.DataFrame(abs(self.reg_model.coef_),columns = [col_name] , index= self.factor_name)
            elif self.method in ['ridge','lasso','ElasticNet','pls']:
                coef = self.reg_model.best_estimator_.coef_.reshape(-1,1)
                var_importance = pd.DataFrame(abs(coef),columns = [col_name] ,index = self.factor_name)
            elif self.method in ['rf','adaBoost','gbm','xgb']:
    #            var_importance = None
                coef = self.reg_model.best_estimator_.feature_importances_.reshape(-1,1)
                var_importance = pd.DataFrame(abs(coef),columns = [col_name] ,index = self.factor_name)
            res = var_importance.sort_values(col_name,ascending = False)
            #对因子重要性进行归一化。
            Dchange = Data_Preprocess.Data_Change('minmax')
            Dchange.fit(res)
            res = Dchange.transform(res)
            #画条形图

            if isplot:

                plt = Data_plot.plot_bar_analysis(res,Top=15)
                plt.title('variable importance')
                plt.show()
            return res
            
class reg_stack():
    '''
    对多个不同的大类模型进行stack
    拟合 - 组合 - 预测
    '''
    def __init__(self,listModelName,isGridSearch = True , dict_para = {},meta_reg = 'linear'):
        
        self.listModelName = listModelName
        self.isGridSearch = isGridSearch
        self.dict_para = dict_para
        self.meta_reg = meta_reg
        #缺省参数
        self.train_model = defaultdict(list)
        self.stack = None
        
    def fit(self,x,y):
        '''
        拟合：
        '''
        x_train = np.array(x)
        y_train = np.array(y).reshape(y.shape[0],)
        model_list = []
        basic_reg = ['linear','ridge','lasso','ElasticNet','pls','svr','knn','dt','rf','adaBoost','gbm','xgb']
        #添加基础回归模型
        for model_name in self.listModelName:
            if model_name in basic_reg:

                reg = reg_model(model_name,isGridSearch = self.isGridSearch)
                
                if model_name in self.dict_para.keys():
                    #如果用户自定义了参数范围，则对模型参数进行设置
                    reg.set_parameters(self.dict_para[model_name])
                else:
                    pass
                #模型拟合
                reg.fit(x,y)
                model_list.append(reg.reg_model)
                
                self.train_model[model_name] = reg
        
        if self.meta_reg == 'linear' :
            meta_reg = linear_model.LinearRegression()
        elif self.meta_reg == 'ridge' :
            meta_reg = linear_model.Ridge()
        self.stack = StackingRegressor(regressors = model_list,meta_regressor = meta_reg)
        self.stack.fit(x_train,y_train)
    
    def predict(self,x):
        return self.stack.predict(x)
    
    def get_vip(self,stack_method = 'weight',isplot = True):
        res = []
        idx = []
        for i,key in enumerate(self.train_model):
            vip = self.train_model[key].get_vip(isplot = False)
            if vip is not None:
                res.append(vip)
                idx.append(i)
        #不同模型结果融合
        temp = pd.concat(res,axis = 1)
        if stack_method == 'avg':
            res = temp.mean(axis = 1).sort_values()
        elif stack_method == 'weight':
            res = np.dot(temp.values,self.stack.coef_[idx])
            res = pd.DataFrame(res,index = temp.index,columns = ['variable importance']).sort_values('variable importance')
        
        #画条形图
        if isplot:
            plt = Data_plot.plot_bar_analysis(res)
            plt.title('variable importance')
            plt.show()
        
        return res
    
class reg_stack_muti():
    '''
    对多个相同子类模型，不同的大类模型进行stack
        抽样 - 筛选 - 拟合 - 组合 - 预测
    '''
    def __init__(self,listModelName,isGridSearch = True , dict_para = {},n_model = 1,
                 benchmark = 'all',boostrap = True,ratioFactorSampling = 1,KPI = 'mse',threshold = None,TopN = 3,stack_method = 'avg'):
        self.listModelName = listModelName
        self.isGridSearch = isGridSearch
        self.dict_para = dict_para
        self.n_model = n_model
        self.benchmark = benchmark
        self.boostrap = boostrap
        self.ratioFactorSampling = ratioFactorSampling
        self.KPI = KPI
        self.threshold = threshold
        self.TopN = TopN
        self.stack_method = stack_method
        
        #缺省参数
        self.train_model = defaultdict(list)
        self.mse_list = []
        
    def sampling(self,x,y):
        '''
        抽样：
        样本抽样：必须，默认对全部样本进行重采样
        因子抽样：可选，默认不抽因子
        '''
        if self.ratioFactorSampling == 1:#不筛选因子
            res_x ,res_y = reg_sample_balance(x,y,benchmark = self.benchmark,boostrap = self.boostrap)
        elif  self.ratioFactorSampling > 0 and self.ratioFactorSampling < 1: #按比例筛选因子
            res_x ,res_y = reg_sample_balance(x,y,benchmark = self.benchmark,boostrap = self.boostrap)
            res_x = res_x.sample(res_x.shape[1] * self.ratioFactorSampling,axis = 1)
        return res_x,res_y
    
    def filtrate(self,model_list,x,y):
        '''
        筛选：
        根据Top_N 筛选
        根据min_mse,min_r2筛选
        '''
        res = []
        if self.threshold is None:#按照TopN筛选
            if len(model_list) < self.TopN:
                res = model_list
            else:
                res_dict = {}
                if self.KPI == 'mse': 
                    for i,reg in enumerate(model_list):
                        res_dict[i] = mean_squared_error(y,reg.predict(x))
                    res_dict = pd.DataFrame(res_dict,index = ['mse']).T.sort_values('mse').iloc[:self.TopN,:]
                elif self.KPI == 'r2':
                    for i,reg in enumerate(model_list):
                        res_dict[i] = r2_score(y,reg.predict(x))
                    res_dict = pd.DataFrame(res_dict,index = ['mse']).T.sort_values('r2').iloc[:self.TopN,:]
                for reg_idx in res_dict.index:
                    res.append(model_list[reg_idx])
                    
        else:#按照阈值筛选
            for reg in model_list:
                if self.KPI == 'mse':
                    kpi = mean_squared_error(y,reg.predict(x))
                    if kpi < self.threshold:
                        res.append(reg)
                        
                elif self.KPI == 'r2':
                    kpi = r2_score(y,reg.predict(x))
                    if kpi > self.threshold:
                        res.append(reg)
        return res
        
    def fit(self,x,y):
        '''
        拟合：
        '''
        basic_reg = ['linear','ridge','lasso','ElasticNet','pls','svr','knn','dt','rf','adaBoost','gbm','xgb']
        X_train, X_test, y_train, y_test = split_train_test(x,y,test_size=0.1)
        for model_name in self.listModelName:
            if model_name in basic_reg:
                #模型预训练
                print('当前训练模型： {} '.format(model_name))
                start =time.clock()
                pre_reg = reg_model(model_name,isGridSearch = self.isGridSearch)
                
                if model_name in self.dict_para.keys():
                    #如果用户自定义了参数范围，则对模型参数进行设置
                    pre_reg.set_parameters(self.dict_para[model_name])
                else:
                    pass
                pre_reg.fit(x,y)
                best_reg_para = pre_reg.best_parameters
                end = time.clock()
                print('优化参数模型，耗时： {} s '.format(end - start))
                start = end
                
                model_list = []
                for i in range(self.n_model):
                #一共生成n_model个模型，先进行样本抽样 
                    
                    X_train_sampling , y_train_sampling = self.sampling(X_train,y_train)
                    reg = reg_model(model_name,isGridSearch = self.isGridSearch)
                    reg.set_parameters(best_reg_para)
                    #模型拟合
                    reg.fit(X_train_sampling,y_train_sampling)
                    model_list.append(reg)
                    end = time.clock()
                    print('第  {} 个模型，耗时： {} s '.format(i+1,end - start))
                    start = end
                    
                #根据筛选条件
                print('进行模型筛选 {} of {} '.format(self.TopN,self.n_model))
                self.train_model[model_name] = self.filtrate(model_list,X_test,y_test)
                
                #计算验证集mse
                sub_model_res = []
                for sub_model in self.train_model[model_name]:
                    sub_model_res.append(pd.DataFrame(sub_model.predict(X_test)))
                sub_model_res = pd.concat(sub_model_res,axis = 1).mean(axis = 1)
                self.mse_list.append(-mean_squared_error(y_test,sub_model_res))
    
    def predict(self,x):
        '''
        预测：
        '''
        res = []
        for model_name in self.listModelName:
            sub_model_res = []
            for sub_model in self.train_model[model_name]:
                sub_model_res.append(pd.DataFrame(sub_model.predict(x)))
            #子模型结果融合
            sub_model_res = pd.concat(sub_model_res,axis = 1).mean(axis = 1)
            res.append(sub_model_res)
        #不同模型结果融合
        if self.stack_method == 'avg':
            res = pd.concat(res,axis = 1).mean(axis = 1)
        elif self.stack_method == 'weight':
            res = pd.concat(res,axis = 1).values
            #对mse进行归一化
            mse = pd.DataFrame(self.mse_list)
            Dchange = Data_Preprocess.Data_Change('minmax')
            mse = Dchange.fit_transform(mse)
            weight = np.array(mse).reshape(len(res),1)
            res = np.dot(res,weight)
        return res.values
    
    def get_vip(self,stack_method = 'avg',isplot = True):
        '''
        计算融合 关键因子
        ‘avg’:对关键因子权重求平均
        ‘weight’:对关键因子权重加权求和
        '''
        res = []
        idx = []
        for i,model_name in enumerate(self.listModelName):
            sub_model_res = []
            for sub_model in self.train_model[model_name]:
                vip = sub_model.get_vip(isplot = False)
                if vip is not None:
                    sub_model_res.append(vip)
                    
            #子模型结果融合
            if len(sub_model_res):
#                factor_name = sub_model_res.index
                idx.append(i)
                sub_model_res = pd.concat(sub_model_res,axis = 1).mean(axis = 1)
                res.append(sub_model_res)
                
        #不同模型结果融合
        if stack_method == 'avg':
            res = pd.concat(res,axis = 1).mean(axis = 1)
        elif stack_method == 'weight':
            res = pd.concat(res,axis = 1).values
            weight = np.array(self.mse_list[idx]).reshape(len(res),1)/sum(self.mse_list)
            res = np.dot(res,weight)

        res = pd.DataFrame(res.values,index = res.index,columns = ['variable importance']).sort_values('variable importance')
        
        #画条形图
        if isplot:
            plt = Data_plot.plot_bar_analysis(res)
            plt.title('variable importance')
            plt.show()
        
        return res
          
def reg_score(reg_input,train_x,train_y,valid_x,valid_y,label = None,is_plot = True,
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
#    pdb.set_trace()
    if y_change is None:
        train_pred_y = reg_input.predict(train_x_input)
    else:
        train_pred_y = y_change.change_back(pd.DataFrame(reg_input.predict(train_x_input),columns=['train_pred_y'],index = train_y.index))
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
        plt = Data_plot.plot_scatter(plot_train_data,issns=False) 
        line_data = np.array([[plot_train_data.max()[0],plot_train_data.max()[0]],[plot_train_data.min()[0],plot_train_data.min()[0]]])
        plt = Data_plot.plot_line(pd.DataFrame(line_data,columns=['y_true','y_pred']))
        plt.show()

    print('训练集：mse = {} , r2 = {}'.format(train_mse,train_r2))
    
    if y_change is None:
        valid_pred_y = reg_input.predict(valid_x_input)
    else:
        valid_pred_y = y_change.change_back(pd.DataFrame(reg_input.predict(valid_x_input),columns=['valid_pred_y'],index = valid_y.index))
        valid_y = y_change.change_back(valid_y)
    valid_mse = mean_squared_error(valid_y,valid_pred_y)
    valid_r2 = r2_score(valid_y,valid_pred_y)
    valid_pred_y = pd.DataFrame(valid_pred_y,columns=['valid_pred_y'],index = valid_y.index)
    
    if is_plot:
        plt = Data_plot.plot_scatter(valid_y)
        plt = Data_plot.plot_line(valid_pred_y,c=['r--'])
        plt.show()
        
        
        plot_valid_data = pd.concat([valid_y,valid_pred_y],axis=1)
        plt = Data_plot.plot_scatter(plot_valid_data,issns=False) 
        line_data = np.array([[plot_valid_data.max()[0],plot_valid_data.max()[0]],[plot_valid_data.min()[0],plot_valid_data.min()[0]]])
        plt = Data_plot.plot_line(pd.DataFrame(line_data,columns=['y_true','y_pred']),)
        plt.show()
    print('验证集：mse = {} , r2 = {}'.format(valid_mse,valid_r2))
    return valid_mse,valid_r2    
    

def cls_sample_balance(x,y,Multiple = 3,boostrap = False,benchmark = 'min'):
    '''
    样本均衡：
    通过筛选，使得正负样本尽量均衡。保证各类样本比例不超过3:1,by label 。
    method = 'random':如果样本失衡，则在多数类别中，进行随机抽取。
    Multiple:大类样本与小类样本数量比
    boostrap：是否重采样
    benchmark：max :以大类为基准，下采样 ，min : 以小类为基准，上采样
    '''

    #统计y的count()
    cnt = Counter(y.iloc[:,0])

    x_res = pd.DataFrame()
    y_res = pd.DataFrame()
    if benchmark == 'min':
        #计算最小的类和类的个数(上采样)
        min_num = min(cnt.values())

        for key in cnt.keys():
            if cnt[key] < min_num * Multiple:
                idx = y[y==key].dropna().index
            else:
                idx = y[y==key].dropna().sample(int(min_num * Multiple),replace=boostrap).index
                    
            x_new = x.loc[idx,:]
            y_new = y.loc[idx,:]
            x_res = pd.concat([x_res,x_new])
            y_res = pd.concat([y_res,y_new])
               
    elif benchmark == 'max': 
        #计算最大的类和类的个数（下采样）
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
            
    elif benchmark == 'all':
        #对全部样本进行采样，只能进行重采样
        idx = y.dropna().sample(len(y),replace=True).index
        
        x_res = x.loc[idx,:]
        y_res = y.loc[idx,:]
        
    return x_res,y_res

class cls_model():
    def __init__(self,method,isGridSearch = True):
        #设置基本参数
        self.method = method
        self.isGridSearch = isGridSearch
        #设置缺省参数
        self.cls_model = None
        self.parameters = None
        self.factor_name = None
        self.best_parameters = None
    
    def set_parameters(self,parameters = None):
        #设置模型参数:如果需要调参，则自定义的是多组参数调参范围，如果不需要调参，则自定义一组参数即可
        if parameters is None: #用户不传参数，则使用默认参数
            if self.isGridSearch == True:
                if self.method == 'logistic':
                    self.method == {'penalty':['l1','l2'],
                                    'C':[0.1,1,10]}
                elif self.method == 'knn':
                    self.parameters = {"n_neighbors": [3,4,5,6,7],
                                       "weights":['uniform','distance'],
                                       'algorithm':['auto','ball_tree','kd_tree','brute']
                                       }
                elif self.method == 'svm':
                    self.parameters = {"C": [0.1,1,10,100],
                                       }
                elif self.method == 'dt':
                    self.parameters ={'max_depth' :[3,5,7]}
                elif self.method == 'rf':
                    self.parameters ={"max_depth": [3, 5, 7],
                                      "n_estimators": [300,500,1000],}
                elif self.method == 'adaBoost':
                    self.parameters ={ "learning_rate": [0.01, 0.1],
                                      "n_estimators": [500,1000],}
                elif self.method == 'gbm':
                    self.parameters ={"max_depth": [3, 5],
                                      "learning_rate": [0.01, 0.1],
                                      "n_estimators": [500,1000],}
                elif self.method == 'xgb':
                    self.parameters ={"max_depth": [3, 5],
                                      "learning_rate": [0.01, 0.1],
                                      "n_estimators": [500,1000],}
                elif self.method == 'bp':
                    self.parameters = {'activation':['logistic','tanh','relu'],
                                       'hidden_layer_sizes' : [(10,),(20,),(100,)],
                                       'max_iter': [200000],}
            else:
                if self.method == 'logistic':
                    self.method == {'penalty':['l2'],
                                    'C':[1]}
                elif self.method == 'knn':
                    self.parameters = {"n_neighbors": [5],}
                elif self.method == 'svm':
                    self.parameters = {"C": [1],
                                       }
                elif self.method == 'dt':
                    self.parameters ={'max_depth' :[5]}
                elif self.method == 'rf':
                    self.parameters ={"max_depth": [5],
                                      "n_estimators": [500],}
                elif self.method == 'adaBoost':
                    self.parameters ={ "learning_rate": [0.1],
                                      "n_estimators": [500],}
                elif self.method == 'gbm':
                    self.parameters ={"max_depth": [5],
                                      "learning_rate": [0.1],
                                      "n_estimators": [500],}
                elif self.method == 'xgb':
                    self.parameters ={"max_depth": [5],
                                      "learning_rate": [0.1],
                                      "n_estimators": [500],}
                elif self.method == 'bp':
                    self.parameters = {'activation':['logistic'],
                                       'hidden_layer_sizes' : [(10,)],
                                       'max_iter': [200000],}
                    
        else:#用户传参数，则以用户参数为准
            self.parameters = parameters
    
    def fit(self,x,y):
        x_train = np.array(x)
        y_train = np.array(y).reshape(len(y))
        self.factor_name = list(x.columns)
#        
        if self.parameters is None:
            self.set_parameters()
        
        scoring = {#"roc": make_scorer(roc_auc_score),
                   "acc": make_scorer(accuracy_score),}
        
        if self.method == 'logistic':
            self.cls_model = linear_model.LogisticRegression()
            self.cls_model.fit(x_train,y_train)
            
        elif self.method == 'knn':
            self.cls_model = GridSearchCV(KNeighborsClassifier(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='acc')
            self.cls_model.fit(x_train,y_train)
        
        elif self.method == 'svm':
            self.cls_model = GridSearchCV(svm.SVC(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='acc')
            self.cls_model.fit(x_train,y_train)
        elif self.method == 'dt':
            self.cls_model = GridSearchCV(tree.DecisionTreeClassifier(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='acc')
            self.cls_model.fit(x_train,y_train)
            
        elif self.method == 'rf':
            self.cls_model = GridSearchCV(esb.RandomForestClassifier(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='acc')
            self.cls_model.fit(x_train,y_train)
            
        elif self.method == 'adaBoost':
            self.cls_model = GridSearchCV(esb.AdaBoostClassifier(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='acc')
            self.cls_model.fit(x_train,y_train)
            
        elif self.method == 'gbm':
            self.cls_model = GridSearchCV(esb.GradientBoostingClassifier(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='acc')
            self.cls_model.fit(x_train,y_train)
        
        elif self.method == 'xgb': 
            self.cls_model = GridSearchCV(XGBClassifier(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='acc')
            self.cls_model.fit(x_train,y_train)
            
        elif self.method == 'bp':
            self.cls_model = GridSearchCV(neural_network.MLPClassifier(),param_grid=self.parameters,cv=5,scoring=scoring,refit ='acc')
            self.cls_model.fit(x_train,y_train)
    
    def predict(self,x):
        #模型预测
        x_pred = np.array(x)
        return self.cls_model.predict(x_pred)
    
    def predict_proba(self,x):
        x_pred = np.array(x)
        try:
            res = self.cls_model.predict_proba(x_pred)
        except:
            res = None
        finally:
            return res
    
    def get_vip(self,isplot=True):
        #计算关键因子重要性
        col_name = 'variable importance'
        if self.method in ['knn','dt','svm','bp']:
            res = None
            
        else:
            if self.method in ['logistic'] :
                mean_coef = pd.DataFrame(abs(self.cls_model.coef_)).T.mean(axis=1)
                var_importance = pd.DataFrame(mean_coef.values,index = self.factor_name , columns = [col_name])
#                var_importance = pd.DataFrame(abs(self.cls_model.coef_),index = [col_name] ,columns = self.factor_name)
                
            elif self.method in ['rf','adaBoost','gbm','xgb']:
                coef = self.cls_model.best_estimator_.feature_importances_.reshape(-1,1)
                var_importance = pd.DataFrame(abs(coef),columns = [col_name] ,index = self.factor_name)
            
            res = var_importance.sort_values(col_name)
            #对因子重要性进行归一化。
            Dchange = Data_Preprocess.Data_Change('minmax')
            Dchange.fit(res)
            res = Dchange.transform(res)
            #画条形图
            if isplot:
                plt = Data_plot.plot_bar_analysis(res)
                plt.title('variable importance')
                plt.show()
        
        return res
    
class cls_model_stack():
    def __init__(self,listModelName,isGridSearch = True , dict_para = {},meta_reg = 'logistic'):
        
        self.listModelName = listModelName
        self.isGridSearch = isGridSearch
        self.dict_para = dict_para
        self.meta_reg = meta_reg
        #缺省参数
        self.train_model = defaultdict(list)
        self.stack = None
    
    def fit(self,x,y):
        '''
        拟合：
        '''
        model_list = []
        basic_cls = ['logistic','knn','svm','dt','rf','adaBoost','gbm','xgb']
        for model_name in self.listModelName:
            if model_name in basic_cls:

                cls = cls_model(model_name,isGridSearch = self.isGridSearch)
                
                if model_name in self.dict_para.keys():
                    #如果用户自定义了参数范围，则对模型参数进行设置
                    cls.set_parameters(self.dict_para[model_name])
                else:
                    pass
                #模型拟合
                cls.fit(x,y)
                model_list.append(cls.cls_model)
                
                self.train_model[model_name] = cls
        
        if self.meta_reg == 'logistic':
            meta_cls = linear_model.LogisticRegression()
            
        elif self.meta_reg == 'knn':
            meta_cls = KNeighborsClassifier()
            
        self.stack = StackingClassifier(classifiers = model_list,meta_classifier = meta_cls)
        self.stack.fit(x.values,y.values.reshape(len(y)))
    
    def predict(self,x):
        return self.stack.predict(x)
    
    def get_vip(self,stack_method = 'avg',isplot = True):
        res = []
        idx = []
        for i,key in enumerate(self.train_model):
            vip = self.train_model[key].get_vip(isplot = False)
            if vip is not None:
                res.append(vip)
                idx.append(i)
        #不同模型结果融合
        if len(res) == 0:
            res = None
        else:
            temp = pd.concat(res,axis = 1)
            if stack_method == 'avg':
                res = temp.mean(axis = 1).sort_values()
                res = pd.DataFrame(res,columns = ['variable importance'])
#            elif stack_method == 'weight':
#                pass
#                res = np.dot(temp.values,self.stack.coef_[idx])
#                res = pd.DataFrame(res,index = temp.index,columns = ['variable importance']).sort_values('variable importance')
            
            #画条形图
            if isplot:
                plt = Data_plot.plot_bar_analysis(res)
                plt.title('variable importance')
                plt.show()
            
        return res

    def predict_proba(self,x):
        x_pred = np.array(x)
        try:
            res = self.stack.predict_proba(x_pred)
        except:
            res = None
        finally:
            return res

class cls_model_stack_muti():
    def __init__(self,listModelName,isGridSearch = True , dict_para = {},n_model = 1,Multiple =3,
                 benchmark = 'all',boostrap = True,ratioFactorSampling = 1,KPI = 'roc',threshold = None,TopN = 3,stack_method = 'avg'):
        self.listModelName = listModelName
        self.isGridSearch = isGridSearch
        self.dict_para = dict_para
        self.n_model = n_model
        self.benchmark = benchmark
        self.boostrap = boostrap
        self.ratioFactorSampling = ratioFactorSampling
        self.KPI = KPI
        self.threshold = threshold
        self.TopN = TopN
        self.stack_method = stack_method
        self.Multiple = Multiple
        
        #缺省参数
        self.train_model = defaultdict(list)
        self.acc_list = []
        
    def sampling(self,x,y):
        '''
        抽样：
        样本抽样：必须，默认对全部样本进行重采样
        因子抽样：可选，默认不抽因子
        '''
        if self.ratioFactorSampling == 1:#不筛选因子
            res_x ,res_y = cls_sample_balance(x,y,benchmark = self.benchmark,boostrap = self.boostrap,Multiple = self.Multiple)
        elif  self.ratioFactorSampling > 0 and self.ratioFactorSampling < 1: #按比例筛选因子
            res_x ,res_y = cls_sample_balance(x,y,benchmark = self.benchmark,boostrap = self.boostrap,Multiple = self.Multiple)
            res_x = res_x.sample(res_x.shape[1] * self.ratioFactorSampling,axis = 1)
        return res_x,res_y
    
    def filtrate(self,model_list,x,y):
        '''
        筛选：
        根据Top_N 筛选
        根据min_mse,min_r2筛选
        '''
        res = []
        if self.threshold is None:#按照TopN筛选
            if len(model_list) < self.TopN:
                res = model_list
            else:
                res_dict = {}
                
                for i,cls in enumerate(model_list):
                    res_dict[i] = accuracy_score(y,cls.predict(x))
                res_dict = pd.DataFrame(res_dict,index = ['acc']).T.sort_values('acc').iloc[:self.TopN,:]
                
                for cls_idx in res_dict.index:
                    res.append(model_list[cls_idx])
                    
        else:#按照阈值筛选
            for cls in model_list:
                if self.KPI == 'roc':
                    kpi = roc_auc_score(y,cls.predict(x))
                    if kpi < self.threshold:
                        res.append(cls)
                        
                elif self.KPI == 'acc':
                    kpi = accuracy_score(y,cls.predict(x))
                    if kpi > self.threshold:
                        res.append(cls)
        return res
        
    def fit(self,x,y):
        '''
        拟合：
        '''
        basic_cls = ['logistic','knn','svm','dt','rf','adaBoost','gbm','xgb','bp']
        X_train, X_test, y_train, y_test = split_train_test(x,y,test_size=0.1)
        for model_name in self.listModelName:
            if model_name in basic_cls:
                
                #模型预训练
                print('当前训练模型： {} '.format(model_name))
                start =time.clock()
                
                pre_cls = cls_model(model_name,isGridSearch = self.isGridSearch)
                
                if model_name in self.dict_para.keys():
                    #如果用户自定义了参数范围，则对模型参数进行设置
                    pre_cls.set_parameters(self.dict_para[model_name])
                else:
                    pass
                pre_cls.fit(x,y)
                best_cls_para = pre_cls.best_parameters
                
                end = time.clock()
                print('优化参数模型，耗时： {} s '.format(end - start))
                start = end
                
                model_list = []
                for i in range(self.n_model):
                #一共生成n_model个模型，先进行样本抽样 
                    X_train_sampling , y_train_sampling = self.sampling(X_train,y_train)
                    cls = cls_model(model_name,isGridSearch = self.isGridSearch)
                    cls.set_parameters(best_cls_para)
                    #模型拟合
                    cls.fit(X_train_sampling,y_train_sampling)
                    model_list.append(cls) 
                    end = time.clock()
                    print('第  {} 个模型，耗时： {} s '.format(i+1,end - start))
                    start = end
                #根据筛选条件
                print('进行模型筛选 {} of {} '.format(self.TopN,self.n_model))
                self.train_model[model_name] = self.filtrate(model_list,X_test,y_test)
                 
                #计算验证集acc
                sub_model_res = []
                for sub_model in self.train_model[model_name]:
                    sub_model_res.append(pd.DataFrame(sub_model.predict(X_test)))
                
                sub_model_res = pd.concat(sub_model_res,axis = 1)
                res_list = []
                for idx in range(len(sub_model_res)):
                    res_list.append(sub_model_res.iloc[idx].value_counts().sort_values().index[-1])
                res = np.array(res_list)
#                sub_model_res = pd.concat(sub_model_res,axis = 1).mean(axis = 1)
                
                self.acc_list.append(accuracy_score(y_test,res))
    
    def predict(self,x):
        '''
        预测：
        '''
        res = []
        model_res = []
        for model_name in self.listModelName:
#            sub_model_res = []
            for sub_model in self.train_model[model_name]:
                model_res.append(pd.DataFrame(sub_model.predict(x)))

        #模型结果投票融合
        model_res = pd.concat(model_res,axis = 1)
        for idx in range(len(model_res)):
            res.append(model_res.iloc[idx].value_counts().sort_values().index[-1])
        res = np.array(res)
        return res
    
    def get_vip(self,isplot =True):
        '''
        计算融合 关键因子
        ‘avg’:对关键因子权重求平均
        ‘weight’:对关键因子权重加权求和
        '''
        
        res = []

        for key in self.train_model.keys():
            if len(self.train_model[key]):
                for i in range(len(self.train_model[key])):     
                    vip = self.train_model[key][i].get_vip(isplot = False)
                    if vip is not None:
                        res.append(vip)
        #不同模型结果融合
        if len(res):
            if self.stack_method == 'avg':
                res = pd.concat(res,axis = 1).mean(axis = 1).sort_values()
            
            res = pd.DataFrame(res,columns = ['variable importance'])
            if isplot:
                plt = Data_plot.plot_bar_analysis(res)
                plt.title('variable importance')
                plt.show()
        else:
            res = None 
        return res
    
    def predict_proba(self,x):
        res = None
        cnt = 0
        x_pred = np.array(x)
        for model_name in self.listModelName:
            for sub_model in self.train_model[model_name]:
                prod_mat = sub_model.predict_proba(x_pred)
                if prod_mat is not None:
                    if res is None:
                        res = prod_mat
                        cnt += 1
                    else:
                        res = res + prod_mat
                        cnt += 1
        if res is not None:
            res = res/cnt
            return res
        else:
            return None
                    
        
def cls_scors(cls,train_x,train_y,valid_x,valid_y,label = None):
    '''
    对分类模型进行评价
    '''
    print('----- 分类模型评分 -----')
    if label is not None:
        train_x_input = train_x.drop(label,axis = 1)
        valid_x_input = valid_x.drop(label,axis = 1)
    else:
        train_x_input = train_x
        valid_x_input = valid_x
        
    train_pred_y = cls.predict(train_x_input)
    train_acc = accuracy_score(train_y,train_pred_y)
    
    n_class = train_y.drop_duplicates().shape[0]
    # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
    y_pred = cls.predict(train_x)
    y_score = cls.predict_proba(train_x)
    y_one_hot = label_binarize(train_y, np.arange(n_class))
    try:
        if n_class > 2:
            if y_score is not None:
                train_auc = roc_auc_score(y_one_hot, y_score, average='micro')
            else:
                train_auc = -1
        else:
            train_auc = roc_auc_score(train_y,y_pred)
    except:
        train_auc = -1
    train_pred_y = pd.DataFrame(train_pred_y,columns=['train_pred_y'],index = train_y.index)
    
    plt = Data_plot.plot_confusion_matrix(train_y,train_pred_y)   
    plt.show()
    print('训练集：acc = {} , auc = {}'.format(train_acc,train_auc))
    
    valid_pred_y = cls.predict(valid_x_input)
    valid_acc = accuracy_score(valid_y,valid_pred_y)
    
    # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
    y_valid_pred = cls.predict(valid_x)
    y_valid_score = cls.predict_proba(valid_x)
    y_valid_one_hot = label_binarize(valid_y, np.arange(n_class))
    
    try:
        if n_class > 2:
            if y_score is not None:
                valid_auc = roc_auc_score(y_valid_one_hot,y_valid_score)
            else:
                valid_auc = -1
        else:
            valid_auc = roc_auc_score(valid_y,y_valid_pred)    
    except:
        valid_auc = -1        
            
    valid_pred_y = pd.DataFrame(valid_pred_y,columns=['valid_pred_y'],index = valid_y.index)
    
    plt = Data_plot.plot_confusion_matrix(valid_y,valid_pred_y)   
    plt.show()
    print('验证集：acc = {} , auc = {}'.format(valid_acc,valid_auc))

class clu_model():
    def __init__(self,method = 'kmeans'):
        self.method = method
        self.clu_model = None
        self.para = None
        
    def fit(self,x,para = None):
        if self.method == 'kmeans':
            self.clu_model = KMeans(para)
            self.para = para
            self.clu_model.fit(x)
             
        elif self.method == 'DBSCAN':#密度聚类
            self.clu_model = DBSCAN(para)
            self.para = para
            self.clu_model.fit(x)
        
        elif self.method =='Agg':#凝聚聚类
            self.clu_model = AgglomerativeClustering(para)
            self.para = para
            self.clu_model.fit(x)
            
    def predict(self,x):
        return self.clu_model.predict(x)
    
    
class pca_clu_reg():
    '''
    实现降维，聚类，再回归的方式
    '''
    def __init__(self,para):
        #进入模型的维度
        self.dim = para['dim'] 
        #降维的模型，PCA
        self.dr_model = None
        #聚类类数,默认kmeans
        self.clu_model = None
        self.n_clu = para['n_clu']
        
        self.method = para['method']
        self.is_stack = para['isstack']
        self.reg_model = {}
 
        
    def fit(self,x,y):
        #降维
        self.dr_model = Data_feature_reduction.Feature_Reduction(n_comp = self.dim)
        self.dr_model.fit(x)
        
        x_dr = self.dr_model.transform(x).iloc[:,0:2]
        
        #聚类
        self.clu_model = clu_model()
        self.clu_model.fit(x_dr,self.n_clu)
        clu_label = self.clu_model.predict(x_dr)  
        clu_label = pd.DataFrame(clu_label,columns = ['label'],index = x.index)
        
        #每一类，进行回归拟合
        for i in range(self.n_clu):            
            x_clu = x.loc[clu_label[clu_label==1].dropna().index,:]
            y_clu = y.loc[clu_label[clu_label==1].dropna().index,:]
            #创建模型
            if self.is_stack:
                reg = reg_stack_muti(self.method,n_model=3,TopN=1)
                reg.fit(x_clu,y_clu)
            else:
                reg = reg_model(self.method)
                reg.fit(x_clu,y_clu)
            self.reg_model[i] = reg

            
    def predict(self,x):
        #降维
        x_dr = self.dr_model.transform(x)

        #聚类
        clu_label = self.clu_model.predict(x_dr.iloc[:,[0,1]])
        clu_label = pd.DataFrame(clu_label,columns = ['label'],index = x.index)
        
#        clu_data = pd.concat([x_dr,clu_label],axis = 1)
        res = []
        for i in range(self.n_clu):
            x_clu = x.loc[clu_label[clu_label==i].dropna().index,:]
            if len(x_clu):
                reg_model = self.reg_model[i]
                y_pred = reg_model.predict(x_clu)
                y_pred = pd.DataFrame(y_pred,index = x_clu.index)
                res.append(y_pred)
        res = pd.concat(res).sort_index()
        res = res.rename(columns = {res.columns[0]:'y_pred'})
        
        return np.array(res)
            

    

                
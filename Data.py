# -*- coding: utf-8 -*-
"""

"""
import pdb
import copy
import numpy as np
import pandas as pd
from Analysis_Tool import Data_Preprocess,Data_feature_reduction,Data_model_building
from sklearn.pipeline import Pipeline

class Data():
    def __init__(self,data):
        self.orgin_data = data
        self.data = copy.copy(self.orgin_data)
        
        #初始化 datatype,默认 1：x, 2: y,3: primary_key,4: secondary_key,初始默认全部为x
        self.data_type = pd.DataFrame(np.ones(data.shape[1]),index = data.columns,columns = ['data_type'])
        
        #初始化主键，次主键
        self.primary_key = None
        self.secondary_key = None
        
        #初始化x,y,默认输入的数据全部是x
        self.x_col = self.data.columns
        self.y_col = None
        self.x = self.setting_data_label(self.x_col,key_type = 'x')
        self.y = None
        
        #初始化idx,idy
        self.idx = None
        self.idy = None
        
        #初始化pipeline_list
        self.Pipeline = None
        self.Pipeline_list = []
        
        #初始化数据转换模型
        self.data_change_model = None
        self.data_feature_reduction_model = None
        
    def get_primary_key(self):
        return list(self.data_type[self.data_type==3].dropna().index)
    
    def get_secondary_key(self):
        return list(self.data_type[self.data_type==4].dropna().index)
    
    def get_x(self):
        return list(self.data_type[self.data_type==1].dropna().index)
    
    def get_y(self):
        return list(self.data_type[self.data_type==2].dropna().index)
        
    def setting_data_idxy(self,idx = '*' , idy = '*'):
        
        if idx == '*':
            self.idx = list(self.orgin_data.index)
        else:
            self.idx = idx
            self.data = self.orgin_data.loc[self.idx,:]
            
        if idy == '*':
            self.idy = list(self.orgin_data.col)
        else:
            self.idy = idy
            self.data = self.orgin_data.loc[:,self.idy]
        
    def setting_data_label(self,col_list,key_type  = 'pri'):
        if key_type == 'pri':
            if len(col_list) == 1:
                #重复性判断
                if len(self.data.loc[:,col_list].drop_duplicates()) == self.data.shape[0]:    
                    self.data_type.loc[col_list,:] = 3
                    print('设置主键成功')
                else:
                    print('主键不能有重复')
            else:
                print('主键只能设一个')
                
        elif key_type == 'sec':
            self.data_type.loc[col_list,:] = 4
            print('设置次级主键成功')
        
        elif key_type =='x': 
#            pdb.set_trace()
            self.data_type.loc[col_list,:] = 1
            print('设置 x 成功')
            
        elif key_type =='y': 
            self.data_type.loc[col_list,:] = 2
            print('设置 y 成功')
        
        #重设x,y
        self.x_col = self.get_x()
        self.x = self.data.loc[:,self.x_col]
        self.y_col = self.get_y()
        self.y = self.data.loc[:,self.y_col]
        primary_key = self.get_primary_key()
        self.primary_key = self.data.loc[:,primary_key]
        secondary_key = self.get_secondary_key()
        self.secondary_key = self.data.loc[:,secondary_key]
        
        
    def get_data_type(self):
        return Data_Preprocess.get_variable_type(self.data)
    
    def get_nan_map(self):
        return self.data.isnull()
    
    def fillnan(self,method = 1):
        fillna = Data_Preprocess.Fillna(method = method)
        self.x = fillna.fit(self.x)
        self.data.loc[:,self.x.columns] = self.x
        #新增工序list
        self.Pipeline_list.append(('fillnan',fillna))
    
    def check_outlier(self,method = '3sigma',muti = 3):
        cek_out = Data_Preprocess.Check_Outlier(method  = method,muti = muti)
        self.x = cek_out.fit(self.x)
        self.data.loc[:,self.x.columns] = self.x
        #新增工序list
        self.Pipeline_list.append(('check_outlier',cek_out))
    
    def data_change(self,method ='minmax'):
        Dchange = Data_Preprocess.Data_Change(method = method)
        Dchange.fit(self.x)
        self.x = Dchange.transform(self.x)
        self.data.loc[:,self.x.columns] = self.x
        #新增工序list
        self.Pipeline_list.append(('data_change',Dchange))
        self.data_change_model = Dchange
        
    def feature_reduction(self,n_comp,method = 'pca'):
        fr = Data_feature_reduction.Feature_Reduction(n_comp,method = method)
        if self.data_change is None:
            self.data_change(method = 'avgstd')
            fr.fit(self.x)
            self.x = fr.transform(self.x)
            
        else:    
            fr.fit(self.x)
            self.x = fr.transform(self.x)
        #新增工序list
        self.Pipeline_list.append(('feature_reduction',fr))
        self.data_feature_reduction_model = fr
    
    def reg_fit(self,reg_model = ['linear'],isGridSearch=True,parameter =  None ,n_model = 3,TopN = 2,reg_type = 'single'):
        X_train, X_test, y_train, y_test = Data_model_building.split_train_test(self.x,self.y)
        if reg_type == 'single':
            reg = Data_model_building.reg_model(reg_model[0],isGridSearch=isGridSearch)
            if parameter is not None:
                reg.set_parameters(parameters = parameter)
#            pdb.set_trace()
            reg.fit(X_train,y_train)
            Data_model_building.reg_score(reg,X_train,y_train, X_test, y_test)
            
        elif reg_type == 'reg_stack':
            reg = Data_model_building.reg_stack(reg_model,isGridSearch=isGridSearch)
            if parameter is not None:
                reg.set_parameters(parameter = parameter)
            reg.fit(X_train,y_train)
            Data_model_building.reg_score(reg,X_train,y_train, X_test, y_test)
            
        elif reg_type == 'reg_stack_muti':
            reg = Data_model_building.reg_stack_muti(reg_model,isGridSearch=isGridSearch,n_model=n_model,TopN=TopN)
            if parameter is not None:
                reg.set_parameters(parameter = parameter)
            reg.fit(X_train,y_train)
            Data_model_building.reg_score(reg,X_train,y_train, X_test, y_test)
            
        #新增工序list
        self.Pipeline_list.append(('reg',reg))
        self.data_reg_model = reg
        
    def transform(self,data):
        change_data = data.loc[:,self.x_col]
        self.Pipeline = Pipeline(self.Pipeline_list)
        return self.Pipeline.transform(change_data)
    
    def predict(self,data):
        change_data = data.loc[:,self.x_col]
        self.Pipeline = Pipeline(self.Pipeline_list)
        return self.Pipeline.predict(change_data)
        
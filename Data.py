# -*- coding: utf-8 -*-
"""

"""
import copy
import numpy as np
import pandas as pd
from Analysis_Tool import Data_Preprocess,Data_feature_reduction
from sklearn.pipeline import Pipeline

class Data():
    def __init__(self,data):
        self.orgin_data = data
        self.data = copy.copy(self.orgin_data)
        
        #初始化 datatype,默认 1：x, 2: y,3: primary_key,4: secondary_key,初始默认全部为x
        self.data_type = pd.DataFrame(np.ones(x.shape[1]),index = x.columns,columns = 'data_type')
        
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
        
    def get_primary_key(self):
        return list(self.data_type[self.data_type==3].dropna().index)
    
    def get_secondary_key(self):
        return list(self.data_type[self.data_type==4].dropna().index)
        
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
                    self.data_type.loc[col_list,0] == 3
                else:
                    print('主键不能有重复')
            else:
                print('主键只能设一个')
                
        elif key_type == 'sec':
            self.data_type.loc[col_list,0] == 4
        
        elif key_type =='x': 
            self.data_type.loc[col_list,0] == 1
            self.x = self.data.loc[:,self.x_col]
            
        elif key_type =='y': 
            self.data_type.loc[col_list,0] == 2
            self.y = self.data.loc[:,self.y_col]
            
    def get_data_type(self):
        return Data_Preprocess.get_variable_type(self.data)
    
    def get_nan_map(self):
        return np.isnan(self.data)
    
    def fillnan(self,method = 1):
        fillna = Data_Preprocess.Fillna(method = method)
        self.x = fillna.fit(self.x)
        #新增工序list
        self.Pipeline_list.append(('fillnan',fillna))
    
    def check_outlier(self,method = 'sigma',muti = 3):
        cek_out = Data_Preprocess.Check_Outlier(method  = method,muti = muti)
        self.x = cek_out.fit(self.x)
        #新增工序list
        self.Pipeline_list.append(('check_outlier',cek_out))
        
    def feature_reduction(self,method = 'pca'):
        fr = Data_feature_reduction.Feature_Reduction(method = method)
        self.x = fr.fit(self.x)
        #新增工序list
        self.Pipeline_list.append(('feature_reduction',fr))
        
    def predict(self,data):
        self.Pipeline = Pipeline(self.Pipeline_list)
        return self.Pipeline.predict(data)
        
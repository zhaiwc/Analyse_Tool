# -*- coding: utf-8 -*-
"""
数据预处理：
1.空值补值
2.数据映射
3.数据缩放
4.数据转换
"""
import pandas as pd
import numpy as np
import copy
import pdb

from collections import defaultdict,Counter
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn import linear_model,tree
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder


def __check_label(data):
    '''
    筛选数据中的文本列
    '''
    label_col = list(set(data.columns) - set(data.describe().columns))
    return label_col

def check_discrete(data,check_num =10):
    '''
    筛选数据中的离散列，默认unique = 10
    '''
    disct_col = []
    for col in data.columns:
        datacol = data[col].dropna()
        if len(set(datacol)) <check_num:
            disct_col.append(col)
        else:
            pass
    return disct_col

def __check_nan(data,ratio = 0.75):
    '''
    筛选数据中大多数为nan的列,默认nan值比例大于0.25
    '''
    nan_col = []
    lines = data.shape[0]
    for col in data.columns:
        if len(data[col].dropna()) < lines * (1 - ratio):
            nan_col.append(col)
        else:
            pass
    return nan_col
 
def __check_constant(data):
    '''
    筛选数据中的恒定值列，unique = 1 或者标准化后std < 10e-5
    '''
    constant_col = []
    
    for col in data.columns:
        datacol = data[[col]].dropna()
        if len(set(datacol.iloc[:,0])) < 2:
            constant_col.append(col)
#        else:
#            #标准化
#            datacol,scaler = data2avgstd(datacol)
#            if datacol.std()[0]< 10e-5:
#                constant_col.append(col)
    return constant_col

def get_variable_type(data):
    '''
    按列进行判断数据的类型：
    判断顺序： -> (-3)空值 -> (-2)文本 -> (-1)常数 -> (1)离散 -> (2)连续    
    '''
    print('----- 因子标签标注 -----')
    
    collist = list(data.columns)
    res = pd.DataFrame(np.zeros(len(collist)),
                       columns =['Type'],index = collist)
    #判断空值
    nan_col = __check_nan(data)
    collist = list(set(collist) - set(nan_col))
    data = data.loc[:,collist]
    res.loc[nan_col,:] = -3
    #判断文本列
    label_col = __check_label(data) 
    collist = list(set(collist) - set(label_col))
    data = data.loc[:,collist]
    res.loc[label_col,:] = -2
    #判断常数列
    constant_col = __check_constant(data)
    collist = list(set(collist) - set(constant_col))
    data = data.loc[:,collist]
    res.loc[constant_col,:] = -1
    #判断离散列
    disct_col = check_discrete(data)
    collist = list(set(collist) - set(disct_col))
    data = data.loc[:,collist]
    res.loc[disct_col,:] = 1
    res.loc[collist,:] = 2
    print('标注结果：\n 空值列：{}，文本列：{}，常数列：{}，离散列：{}，连续列 {}.'.format(len(nan_col),
          len(label_col),len(constant_col),len(disct_col),len(collist)))
    return res
    


class Fillna():
    def __init__(self,method):
        self.method = method
        self.label_col = None
        self.drop_col = None
        self.mean_mat = {}
        self.median_mat = {}
        self.model_list = {}
        
    def fill_nan(self,data,mean_mat,median_mat):
        if self.method == 1:     
            data = data.fillna(method ='ffill')
        
        elif self.method == 2:
            for col in data.columns:
                if check_discrete(data[[col]]):
                    data.loc[:,[col]] = data[[col]].fillna(median_mat.loc[col])
                else:
                    data.loc[:,[col]] = data[[col]].fillna(mean_mat.loc[col])
                    
        elif self.method == 3:
            #计算相关性
            fillnadata = copy.copy(data)
            fillnadata = fillnadata.fillna(method ='ffill').dropna()
            corr_df =  fillnadata.corr()
            
            for col in data.columns:
                #建立分类模型
                max_corr_col = abs(corr_df[col].drop(col,axis = 0).dropna()).sort_values().index[-1]
                
                train_data = data.loc[:,[max_corr_col,col]].dropna()
                idx_nan = data[np.isnan(data.loc[:,col])].index
                if col in self.model_list.keys():
                    df_model = self.model_list[col]
                else:
                    df_model = KNeighborsRegressor()
                    
                    x = train_data.loc[:,max_corr_col].values.reshape(-1,1)
                    y = train_data.loc[:,col].values.reshape(-1,1)
                    df_model.fit(x,y)
                    self.model_list[col] = df_model
                if len(idx_nan):  
                    data.loc[idx_nan,col] = df_model.predict(fillnadata.loc[idx_nan,max_corr_col].values.reshape(-1,1)) 
        return data
        
    def fit(self,data,label_col = None):
        '''
        拟合
        '''
        print('-----  数据补值  -----')
        print('数据缺失统计信息：')
        lines,cols = data.shape 
        dropna_line = data.dropna(how = 'any',axis = 0).shape[0] 
        dropna_col = data.dropna(how = 'any',axis = 1).shape[1] 
        print('data dimension: {} lines, {} columns. \n nan dimension:'
              ' {} lines, {} columns.'.format(lines,cols,lines - dropna_line,cols - dropna_col))
        
        #drop the columns that almost nan
        drop_col = []
        for col in data.columns:
            if len(data[col].dropna()) < lines * 0.25:
                drop_col.append(col)
        print('the num of column almost(>75%) nan is {}'.format(len(drop_col)))
        print('删除空值列结果：\n原数据 {} 列，其中，空值列： {} 列，数据列： {} 列。 '.format(data.shape[1],len(drop_col),data.shape[1]-len(drop_col))) 
        data = data.drop(drop_col,axis=1)
        self.drop_col = drop_col
        
        
        if self.label_col is None or self.method == 3:
            #计算mean,median
            self.mean_mat['Total'] = data.mean(axis = 0)
            self.median_mat['Total'] = data.median(axis = 0)
            
            #进行补值
            res = self.fill_nan(data,self.mean_mat['Total'],self.median_mat['Total'])
            
        else:
            res = pd.DataFrame()
            for key,group in data.groupby(self.label_col):
                #计算mean,median
                self.mean_mat[key] = group.mean(axis = 0)
                self.median_mat[key] = group.median(axis = 0)
                
                group = self.fill_nan(group,self.mean_mat[key],self.median_mat[key])
                res = pd.concat([res,group])
            
        return res
    
    def transform(self,data):
        '''
        预测
        '''
        if self.drop_col is not None:
            data = data.drop(self.drop_col,axis = 1)
        
        if self.label_col is None:
            res = self.fill_nan(data,self.mean_mat['Total'],self.median_mat['Total'])
        else:
            res = pd.DataFrame()
            for key,group in data.groupby(self.label_col):
                group = self.fill_nan(group,self.mean_mat[key],self.median_mat[key])
                res = pd.concat([res,group])
        return res
    
    def predict(self,data):
        return self.transform(data)
    
class Data_Encoding():
    def __init__(self,method):
        self.method = method
        self.encoding = {}
        self.labelcolumns = None
        
    def fit(self,strdata,columnlist = None ):
        '''
        根据获取的数据,指定列,生成字典映射。
        order:顺序编码
        onehot:01独热编码
        onehotPCA,独热编码后降维
        '''
        
        if columnlist is None:
            self.columnlist = strdata.columns
        else:
            self.columnlist = columnlist
        
        if self.method =='order':
            for col in self.columnlist:
                
                Encoding = preprocessing.LabelEncoder()
                Encoding.fit(strdata[[col]])
                self.encoding[col] = Encoding
            
        elif self.method == 'onehot':#先转order,再转Oht
            for col in self.columnlist:
                self.encoding[col] = {}
                Encoding = preprocessing.LabelEncoder()
                temp = Encoding.fit_transform(strdata[[col]])
                self.encoding[col]['order'] = Encoding
                Oht = preprocessing.OneHotEncoder()
                Oht.fit(temp.reshape(-1,1))
                self.encoding[col]['Oht'] = Oht
                
    
    def inverse_transform(self,codedata):

        strdata =copy.copy(codedata)
        if self.method == 'order':
            for key in self.encoding.keys:
                strdata[[key]] = self.encoding[key][0].inverse_transform(codedata[[key]])

        elif self.method == 'onehot':
            #创建结果集
            for key in self.encoding.keys:
                temp = self.encoding[key]['Oht'].inverse_transform(codedata[[key]])
                strdata[[key]] = self.encoding[key]['order'].inverse_transform(temp)
                            
        return strdata
    
    def transform(self,strdata):
        
        codedata = copy.copy(strdata)
        if self.method == 'order':
            for key in self.encoding.keys():
                codedata[[key]] = self.encoding[key].transform(codedata[[key]]).reshape(len(strdata),1)
        
        elif self.method == 'onehot':
            for key in self.encoding.keys():
                temp = self.encoding[key]['order'].transform(codedata[[key]])
                trans_data = self.encoding[key]['Oht'].transform(temp.reshape(-1,1)).toarray()
                trans_data = pd.DataFrame(trans_data,index = codedata.index)
                codedata = codedata.drop(key,axis=1)
                codedata = pd.concat([codedata,trans_data],axis=1)

        return codedata

class Data_Change():
    '''
    数据变换类
    '''
    def __init__(self,method):
        self.method = method
        self.scaler = None
        self.data_col = None
        self.is_place = True
        
    def fit(self,data,is_replace = True):
        print('----- 因子变换: {} -----'.format(self.method))
        self.is_place = is_replace
        if self.method == 'log':
            self.data_col = data.columns
            for col in self.data_col:
                new_col = pd.DataFrame(np.log(data[[col]]),columns = ['ln_' + col],index = data.index)
            data = pd.concat([data,new_col],axis=1)
            if self.is_place:
                data = data.drop(self.data_col,axis =1)
        elif self.method == 'avgstd':
            scaler = preprocessing.StandardScaler()
            data = pd.DataFrame(scaler.fit_transform(data),
                                index = data.index, columns = data.columns)
            self.scaler = scaler
            
        elif self.method == 'minmax':
            scaler = preprocessing.MinMaxScaler()
            data = pd.DataFrame(scaler.fit_transform(data),
                            index = data.index,columns = data.columns)
            self.scaler = scaler
            
            
        elif self.method == 'poly':
            scaler = preprocessing.PolynomialFeatures()
            scaler.fit(data)
            self.scaler = scaler
        
    def transform(self,data):    
        if self.method == 'log':
            self.data_col = data.columns
            for col in self.data_col:
                new_col = pd.DataFrame(np.log(data[[col]]),columns = ['ln_' + col],index = data.index)
            data = pd.concat([data,new_col],axis=1)
            if self.is_replace:
                data = data.drop(self.data_col,axis =1)
                
        elif self.method == 'avgstd':
            data = pd.DataFrame(self.scaler.transform(data),
                                index = data.index, columns = data.columns)
                
        elif self.method == 'minmax':
            data = pd.DataFrame(self.scaler.transform(data),
                                index = data.index, columns = data.columns)     
        
        elif self.method == 'poly':
            data = pd.DataFrame(self.scaler.transform(data),
                                columns = self.scaler.get_feature_names())
        return data
    
    def inverse_transform(self,data):
        if self.method == 'log':
            data = pd.DataFrame(np.exp(data),columns = self.data_col)
        elif self.method == 'avgstd':
            data = pd.DataFrame(self.scaler.inverse_transform(data),
                       index = data.index,columns =data.columns)
        elif self.method == 'minmax':
            data = pd.DataFrame(self.scaler.inverse_transform(data),
                        index = data.index,columns = data.columns)
        elif self.method == 'poly':
            data = None
        
        return data
        

class Check_Outlier():
    def __init__(self,method = '3sigma',muti = 3):
        self.method = method
        self.muti = 3
        
        #上下限
        self.upline = None
        self.dnline = None
        
    def fit(self,x):
        if self.method == '3sigma':
            self.upline = x.mean() + self.muti * x.std()
            self.dnline = x.mean() - self.muti * x.std()
        elif self.method == 'tukey':
            x = x.fillna(method = 'ffill')
            n25 = pd.DataFrame(np.percentile(x,25,axis=0),index = x.columns)
            n75 = pd.DataFrame(np.percentile(x,75,axis=0),index = x.columns)
            self.upline = n75 + self.muti * ( n75 - n25 )
            self.dnline = n25 - self.muti * ( n75 - n25 )
   
        x[x>self.upline] = np.nan
        x[x<self.dnline] = np.nan
        
        return x
    
    def transform(self,x):
        
        x[x>self.upline] = np.nan
        x[x<self.dnline] = np.nan
        
        return x
        
def split_datecol(data,thoeshold =10e12):
    '''
    根据阈值判断时间列。10e13: yyyymmddHHMMSS
    '''
    date_col = data.mean()[data.mean()>thoeshold].index.tolist()
    col = set(data.columns)-set(date_col)
    print('拆分时间列结果：原数据 {} 列，其中，时间列: {} 列，数据列: {} 列。 '.format(data.shape[1],len(date_col),len(col)))
    date = data.loc[:,date_col]
    data = data.loc[:,col]
        
    return data,date

def drop_constant_col(data,columnlist = None ,label_col = None):
    '''
    删除恒定值的列
    恒值列：
    1.某一列全部为一恒定值。
    2.分label都为一恒定值。
    '''
    if columnlist is None:
        columnlist = list(data.columns)
    if label_col is not None and label_col in columnlist:
        columnlist.remove(label_col)
        
    drop_col = []
    if label_col is None:
        for col in columnlist:
            if len(set(data[col].dropna())) <2:
                drop_col.append(col)
    else:
        for col in columnlist:
            if len(data.loc[:,[col,label_col]].drop_duplicates()) == len(set(data[label_col])):
                drop_col.append(col)
            
    print('删除恒定值结果：原数据 {} 列，其中,恒值列: {} 列，数据列: {} 列。'.format(data.shape[1],len(drop_col),data.shape[1]-len(drop_col)))
    if len(drop_col):
        data = data.drop(drop_col,axis =1)
    return data,drop_col
            
def long2width(data,index,col,value):
    '''
    长表转宽表
    数据库格式长表转宽表，选定index,col,value,生成宽表。
    其他列将根据index去重复后，保留最后一行，再与宽表合并。
    '''
    wid_tabel = data.pivot(index,col,value)
    data = data.drop([col,value],axis=1) 
    data = data.drop_duplicates([index],keep = 'last')
    res = pd.merge(wid_tabel,data,on = index)
    return res
    


    

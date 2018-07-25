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
from sklearn.neighbors import KNeighborsClassifier


def __check_label(data):
    '''
    筛选数据中的文本列
    '''
    label_col = list(set(data.columns) - set(data.describe().columns))
    return label_col

def __check_discrete(data,check_num =10):
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
    print('-----因子标签标注-----')
    
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
    disct_col = __check_discrete(data)
    collist = list(set(collist) - set(disct_col))
    data = data.loc[:,collist]
    res.loc[disct_col,:] = 1
    res.loc[collist,:] = 2
    print('标注结果：\n 空值列：{}，文本列：{}，常数列：{}，离散列：{}，连续列 {}.'.format(len(nan_col),
          len(label_col),len(constant_col),len(disct_col),len(collist)))
    return res
    
def __fill_nan(data,method =1,label_col = None):
    if method == 1:     
        data = data.fillna(method ='ffill')
            
    elif method == 2:
        for col in data.columns:
            if __check_discrete(data[[col]]):
                data.loc[:,[col]] = data[[col]].fillna(data[[col]].median())
            else:
                data.loc[:,[col]] = data[[col]].fillna(data[[col]].mean())
                
    elif method == 3:
        for col in data.columns:
            if len(data[col].dropna()) == len(data[col]):
                continue
            else: 
                pass

            #该columns 为目标col
            if label_col is not None:
                x_label = list(set(data.columns) - set([col]) - set([label_col]))
            else:
                x_label = list(set(data.columns) - set(col))
            #预测序号，训练序号
            pred_idx = data[col][data[col].isnull()].index.tolist()
            train_idx = list(set(data.index.tolist()) - set(pred_idx))
            
            train_x = data.loc[train_idx,x_label]
            train_y = data.loc[train_idx,col]
            pred_x = data.loc[pred_idx,x_label]
            #训练数据和测试数据可能存在nan值，需要剔除取交集后再训练
            train_x = train_x.dropna(how = 'any',axis=1)
            pred_x = pred_x.dropna(how = 'any',axis=1)
            x_label = list(set(train_x.columns)&set(pred_x.columns))
            #重新整理数据
            train_x = train_x.loc[:,x_label]
            pred_x = pred_x.loc[:,x_label]

            if __check_discrete(data[col]):
                try:
                    knn_model = KNeighborsClassifier()
                    knn_model.fit(train_x,train_y)
                    pred_y = knn_model.predict(pred_x)
                    data[col].loc[pred_idx] = pred_y
                except:
                    print('error1:',col)
                    data.loc[:,[col]] = data[[col]].fillna(data[[col]].median())
            else:
                try:
                    Linear_m = linear_model.LinearRegression()
                    Linear_m.fit(train_x,train_y)
                    pred_y = Linear_m.predict(pred_x)
                    data[col].loc[pred_idx] = pred_y
                except:
                    print('error2:',col)
                    data.loc[:,[col]] = data[[col]].fillna(data[[col]].mean())
    return data

def fill_nan(data,label_col = None, method = 1):
    '''
    by 列区分离散和连续。
    method = 1 
        离散：上一个值补值
        连续：上一个值补值
    method = 2
        离散：按众数补值
        连续：按均值补值
    method = 3
        离散：按分类补值
        连续：按回归补值
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
    
    if label_col is None:
        res =  __fill_nan(data,method = method)
        
    else:
        res = pd.DataFrame()
        for key,group in data.groupby(label_col):
            group = __fill_nan(group,method = method,label_col = label_col)
            res = pd.concat([res,group])

    return res,drop_col


class Data_Encoding():
    def __init__(self,method):
        self.method = method
        self.map_dict = defaultdict(dict)
        self.labelcolumns = None
        
    def data2label(self,strdata,columnlist):
        '''
        根据获取的数据,指定列,生成字典映射。
        order:顺序编码
        onehot:01独热编码
        onehotPCA,独热编码后降维
        '''

        
        if self.method =='order':
            codedata =copy.copy(strdata)
            self.labelcolumns = columnlist
            for col in columnlist:
                #进行转换
                data_col_gpby = strdata[[col]].groupby(strdata[col]).count()
                
                for idx in range(len(data_col_gpby)):
                    codedata[[col]] = codedata[[col]].replace(data_col_gpby.index[idx],idx)
                    self.map_dict[col][idx] = data_col_gpby.index[idx]
        elif self.method == 'onehot':
            codedata = []
            self.labelcolumns = columnlist
            for col in columnlist:
                #进行转换
                thscol = pd.get_dummies(strdata[[col]])
                #生成转换mapdict
                drop_dup = thscol.drop_duplicates()
                self.map_dict[col]['col'] = thscol.columns
                for col_sub in thscol.columns:
                    col_list = col_sub.split('_')
                    self.map_dict[col][col_list[1]] = np.array(drop_dup[drop_dup.loc[:,col_sub] == 1])
                
                codedata.append(thscol)
            codedata = pd.concat(codedata,axis=1)

        return codedata
    
    def label2data(self,codedata):

        strdata =copy.copy(codedata)
        if self.method == 'order':
            for key in self.map_dict:
                for code in self.map_dict[key]:
                    strdata[[key]] = strdata[[key]].replace(code,self.map_dict[key][code])
        elif self.method == 'onehot':
            #创建结果集
            res = pd.DataFrame(np.ones((codedata.shape[0],len(self.labelcolumns))))
            cnt = 0
            
            for col in codedata.columns:
                col_list = col.split('_')
                if  col_list[0] == res.columns[0]:
                    pass
                else:
                    res = res.rename(columns = {cnt:col_list[0]})
                    cnt += 1
                #赋值
                res.loc[codedata[codedata.loc[:,col]==1].index,col_list[0]] = col_list[1]
                
            strdata = res
            
        return strdata
    
    def transform(self,strdata,columnlist = None):
        codedata = copy.copy(strdata)
        if self.method == 'order':
            if columnlist is None:
                columnlist = self.labelcolumns
            for col in columnlist:
                for key in self.map_dict[col]:
                    codedata[[col]] = codedata[[col]].replace(key,self.map_dict[col][key])
        
        elif self.method == 'onehot':
            res = []
            if columnlist is None:
                columnlist = self.labelcolumns
            for col in columnlist:
                temp = []
                columns_stand =  self.map_dict[col]['col']
                for key in self.map_dict[col].keys():
                    if key != 'col':
                        idx = strdata[strdata.loc[:,col] == key].index
                        ndarray = np.array(list(self.map_dict[col][key])*len(idx)).reshape(len(idx),-1)
                        temp.append(pd.DataFrame(ndarray,index = idx,columns = columns_stand))
                res.append(pd.concat(temp))
            res = pd.concat(res,axis = 1)
            codedata = res.reindex(strdata.index)
        return codedata

class Data_Change():
    '''
    数据变换类
    '''
    def __init__(self,method):
        self.method = method
        self.scaler = None
        self.data_col = None
        
    def fit_transform(self,data,is_replace = True):
        print('-----开始进行因子变换:转换方法：{}-----'.format(self.method))

            
        if self.method == 'log':
            self.data_col = data.columns
            for col in self.data_col:
                new_col = pd.DataFrame(np.log(data[[col]]),columns = ['ln_' + col],index = data.index)
            data = pd.concat([data,new_col],axis=1)
            if is_replace:
                data = data.drop(self.data_col,axis =1)
                
        elif self.method == 'avgstd':
            if self.scaler is None:
                scaler = preprocessing.StandardScaler()
                data = pd.DataFrame(scaler.fit_transform(data),
                                    index = data.index, columns = data.columns)
                self.scaler = scaler
            else:
                data = pd.DataFrame(self.scaler.fit_transform(data),
                                    index = data.index, columns = data.columns)
                
        elif self.method == 'minmax':
            if self.scaler is None:
                scaler = preprocessing.MinMaxScaler()
                data = pd.DataFrame(scaler.fit_transform(data),
                                index = data.index,columns = data.columns)
                self.scaler = scaler
            else:
                data = pd.DataFrame(self.scaler.fit_transform(data),
                                    index = data.index, columns = data.columns)
            
        return data
    
    def change_back(self,data):
        if self.method == 'log':
            data = pd.DataFrame(np.exp(data),columns = self.data_col)
        elif self.method == 'avgstd':
            data = pd.DataFrame(self.scaler.inverse_transform(data),
                       index = data.index,columns =data.columns)
        elif self.method == 'minmax':
            data = pd.DataFrame(self.scaler.inverse_transform(data),
                        index = data.index,columns = data.columns)
        
        return data
        



#def data_change(data,method,columnlist = None ,is_drop = True,**kw):
#    '''
#    对每一列进行数据变换：
#    1.对数变换
#    2.差分变换
#    3.交叉项变换
#    4.二次项变换
#    5.偏差量变换
#    6.偏离量变换
#    '''
#    if columnlist is None:
#        columnlist = data.columns
#    else:
#        pass
#    
#    print('-----开始进行因子变换:转换方法：{}-----'.format(method))
#    orgin_col = len(data.columns)
#    if method ==3:
#        len_col = len(columnlist)
#        if len_col >=2:
#            for i in range(len_col):
#                for j in range(len_col):
#                    if i < j:
#                        ths_factor = pd.DataFrame(np.array(data[columnlist[i]]) * np.array(data[columnlist[j]]),
#                                                  columns = [str(columnlist[i]) + '*' + str(columnlist[j])],index = data.index)
#                        data = pd.concat([data,ths_factor],axis=1)
#        else:
#            print('计算交互作用至少包含2个因子')
#    else:
#        for col in columnlist:
#            if method == 1:
#                new_col = pd.DataFrame(np.log(data[[col]]),columns = ['ln_' + col],index = data.index)
#            elif method == 2:
#                if 'para' in kw:
#                    new_col = pd.DataFrame(data[[col]].diff(kw['para'][0]),columns = ['diff_' + col],index = data.index)
#                else:
#                    new_col = pd.DataFrame(data[[col]].diff(),columns =['diff_' + col],index = data.index)
#            elif method == 4:
#                new_col = pd.DataFrame(np.array(data[col]) *np.array(data[col]),columns=[col + '_2'],index = data.index)
#
#            data = pd.concat([data,new_col],axis=1)
#    final_col = len(data.columns)
#    print('原始列数：{} , 增加列数：{} , 最终列数： {} '.format(orgin_col,final_col - orgin_col,final_col))
#    if is_drop:
#        data = data.drop(columnlist,axis =1)
#    
#    return data

def __check_outlier_one_col(x,method ='3sigma',muti = 3):
    '''
    一列异常数据检验，返回异常的index。
    method = '3sigma': 计算上下3倍std
    method = 'tukey' : Q1 - 1.5*(Q3-Q1) ,Q3 + 1.5*(Q3-Q1)
    '''
    x = x.replace(np.inf,np.nan)
    if method == '3sigma':
        upline = x.mean() + muti * x.std()
        dnline = x.mean() - muti * x.std()
    elif method == 'tukey':
        upline = np.percentile(x,75)  + muti * (np.percentile(x,75) - np.percentile(x,25))
        dnline = np.percentile(x,25)  - muti * (np.percentile(x,75) - np.percentile(x,25))
    outlier_index = list(x[(x>upline)|(x<dnline)].dropna().index)    
#    pct = len(x[(x>upline)|(x<dnline)])/len(x)
    return outlier_index 

def check_outlier(x, method = '3sigma',fill = 'nan',muti = 3):
    '''
    循环每一列，根据method判断该列异常的index,再根据填充方法进行相应的替换
    fill: 'nan' 使用nan值进行替换，'mean'使用均值进行替换.
    '''
    print('-----异常值处理-----')
    print('异常判断方法： {}，异常值处理方法： {}。'.format(method,fill))
#    pct_tot = pd.DataFrame()
    for col in x.columns:
#        print(col)
        x_col = x.loc[:,[col]]
        if fill == 'nan':
            outlier_index = __check_outlier_one_col(x_col,method = method,muti = muti)
            x_col.loc[outlier_index,:] = np.nan
        elif fill == 'mean':
            outlier_index = __check_outlier_one_col(x_col,method = method,muti = muti)
            x_col.loc[outlier_index,:] = x_col.mean()
        x.loc[:,[col]] = x_col
        
#        pctdf = pd.DataFrame([pct,col],index = ['pct','col']).T
#        pct_tot = pd.concat([pct_tot,pctdf])
#    pct_tot.set_index('col')
#    pct_tot = pct_tot.sort_values('pct',ascending ='False')
    
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
    


#def data_balance(x,y,label_col = None,method ='random',rank_by = None ):
#    if label_col is None:
#        x_res,y_res = sample_balance(x,y,method =method,rank_by = rank_by )
#    else:
#        data = pd.concat([])
##        for group in :
#            
            
            
            

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
    
    

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
import pdb

from collections import defaultdict,Counter
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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
        else:
            #标准化
            datacol,scaler = data2avgstd(datacol)
            if datacol.std()[0]< 10e-5:
                constant_col.append(col)
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
    #statistic information
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
    print('删除空值列结果：原数据 {} 列，其中，空值列： {} 列，数据列： {} 列。 '.format(data.shape[1],len(drop_col),data.shape[1]-len(drop_col))) 
    data = data.drop(drop_col,axis=1)
    print('开始对数据进行补值.....')
    if label_col is None:
        res =  __fill_nan(data,method = method)
        
    else:
        res = pd.DataFrame()
        for key,group in data.groupby(label_col):
            group = __fill_nan(group,method = method,label_col = label_col)
            res = pd.concat([res,group])

    return res,drop_col

            
def map_label2code(data,columnlist):
    '''
    根据获取的数据,指定列,生成字典映射。
    data:输入的数据
    columnlist:需要映射的分类
    '''
    
    map_dict = defaultdict(dict)
    for col in columnlist:
        data_col_gpby = data[[col]].groupby(data[col]).count()
        for idx in range(len(data_col_gpby)):
            data[[col]] = data[[col]].replace(data_col_gpby.index[idx],idx)
            map_dict[col][idx] = data_col_gpby.index[idx]       
    return data,map_dict

def map_code2label(data,map_dict):
    '''
    根据获取的数据,映射字典还原。
    data:输入的数据
    map_dict:映射的字典
    '''
    for key in map_dict:
        for code in map_dict[key]:
            data[[key]] = data[[key]].replace(code,map_dict[key][code])
    return data
        
def dummy(data,columnlist):
    '''
    根据指定的离散变量列，全部转换为哑变量列
    '''
    for each in columnlist:
        ths_dummy = pd.get_dummies(data[each],prefix = each,drop_first = False)
        data = pd.concat([data,ths_dummy],axis =1)
    data = data.drop(columnlist,axis =1)
    return data

def one_hot_encode(data):
    '''
    实现独热编码
    '''
    pass

class data_change():
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
        


def data2minmax(data):
    '''
    按最大最小值的方法进行数据映射
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    data = pd.DataFrame(min_max_scaler.fit_transform(data),
                        index = data.index,columns = data.columns)
    return data,min_max_scaler


def minmax2data(data,min_max_scaler):
    '''
    将最大最小值映射过的数据还原
    '''
    data = pd.DataFrame(min_max_scaler.inverse_transform(data),
                        index = data.index,columns = data.columns)
    return data


def data2avgstd(data):
    '''
    按均值标准差的方法进行数据映射
    '''
    scaler = preprocessing.StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data),
                        index = data.index, columns = data.columns)
    return data,scaler


def avgstd2data(data,scaler):
    '''
     按均值标准差映射进行数据还原
    '''
    res = pd.DataFrame(scaler.inverse_transform(data),
                       index = data.index,columns =data.columns)
    return res

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
    
def split_train_test(x,y,test_size = 0.2,random_state = None):
    '''
    拆分数据训练集，测试集
    '''
    X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def sample_balance(x,y,method ='random',rank_by = None,Multiple = 4,boostrap =False,benchmark = 'min'):
    '''
    样本均衡：
    通过样本筛选，使得正负样本尽量均衡。保证各类样本比例不超过4:1,by label 。
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
    
    

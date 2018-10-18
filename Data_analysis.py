# -*- coding: utf-8 -*-
"""
1.描述性统计分析
2.自相关性分析
"""

import pandas as pd 
import numpy as np
import pdb
import copy

from Analysis_Tool.Data_Preprocess import check_discrete
from Analysis_Tool.helper import dcor,CausalCalculator,discrete,continuous
from Analysis_Tool import Data_plot,Data_Preprocess,Data_model_building,Data_feature_reduction
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest

def _describe_data(data):
    describe_df = data.describe()
    skew = pd.DataFrame(data.skew(),columns = ['skew']).T
    kurt = pd.DataFrame(data.kurt(),columns = ['kurt']).T
        
    describe_df = pd.concat([describe_df,skew,kurt])
    return describe_df.T

def describe_analysis(data,columnslist = None,label_col =None):
    '''
    按输入的列计算描述性统计量：
    mean,std,max,min,四分位数,skew,kurt
    画图
    '''

    if columnslist is None:
        columnslist = list(data.columns)
    if label_col is not None and label_col in columnslist:
        columnslist = list(columnslist)
        columnslist.remove(label_col)
        
    print('开始对数据进行描述性统计分析...')
    check_col = _describe_data(data).index
    if label_col is None:
        data = data[columnslist]        
        describe_df = _describe_data(data)
        
    else:
        describe_df = pd.DataFrame()      
        for key,group in data.groupby([label_col]):
            des_df = _describe_data(group)
            index_df = pd.DataFrame([str(x) + '_'+ str(key) for x in des_df.index],
                                     columns=['index'],index = des_df.index)
            des_df = pd.concat([index_df,des_df],axis=1)
            describe_df = pd.concat([describe_df,des_df])
        describe_df = describe_df.set_index('index')
    plt_col = list(set(check_col)&set(columnslist))
    
    plt = Data_plot.plot_describe(data,plt_col,label_col)
    plt.show()
    return describe_df

def spc_analysis(data,p1 = None,p2 = None,method ='3sigma'):
    '''
    spc管控分析
    method = 3sigma/tukey
    针对每一个因子x，计算超出管控线的样本个数
    '''
    res_dict = {}
    #筛选数据列
    num_col = data.describe().columns

    data = data.loc[:,num_col]
    if method == '3sigma':
        for col in data.columns:
            #计算指标
            temp = data.loc[:,col]
            mean_data = temp.mean()
            std_data = temp.std()
            sigma3 = mean_data + 3*std_data
            sigma_3 =  mean_data - 3*std_data
            #计算超出管控线的个数
            res_dict[col] = len(temp[(temp>sigma3) | (temp<sigma_3)])
        res = pd.DataFrame(res_dict,index = ['cnt']).T.sort_values('cnt')
        #画图
        plt = Data_plot.plot_spc(data.loc[:,[res.index[-1]]],method = method)
        plt.show()
    elif method =='tukey':
        for col in data.columns:
            for col in data.columns:
                #计算指标
                temp = data.loc[:,col]
                perc25 = np.percentile(temp,25)
                perc75 = np.percentile(temp,75)
                upper = perc75 + 3 *(perc75 - perc25)
                lower = perc25 - 3 *(perc75 - perc25)
                res_dict[col] = len(temp[(temp>upper) | (temp<lower)])
        res = pd.DataFrame(res_dict,index = ['cnt']).T.sort_values('cnt')
        #画图
        plt = Data_plot.plot_spc(data.loc[:,[res.index[-1]]],method = method)
        plt.show()
    return res

def _autocorr(data,threshold = 0.8,method = 'pearson'):
    drop_col = set()
    if method == 'pearson':
        autocorr_df = data.corr()
    elif method == 'rand':
        autocorr_df = __rand_index_corr(data)
    
    #判断是否需要剔除    
    for idx in range(len(autocorr_df)):
        for idy in range(len(autocorr_df)):
            if idx>=idy:
                continue
            else:
                if data.columns[idx] in drop_col:
                    continue
                else:
                    if abs(autocorr_df.iloc[idx,idy])>=threshold:
                        drop_col.add(data.columns[idy])
                    else:
                        continue
    return autocorr_df,drop_col

def __rand_index_corr(data):
    '''
    兰德指数，计算离散变量之间的相关性,返回相关性矩阵。
    '''
    corrdf = pd.DataFrame(np.zeros([data.shape[1],data.shape[1]]),
                          columns = data.columns,index = data.columns)
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            x = np.array(data.iloc[:,i])
            y = np.array(data.iloc[:,j])
            corrdf.iloc[i,j] = adjusted_rand_score(x,y)
    return corrdf

def autocorr_analysis(data,columnslist = None,label_col =None,
                      threshold = 0.8,isdrop = True,method = 'pearson',isplot = False):
    '''
    自相关性分析:
        求输入矩阵的自相关矩阵，自相关图，根据阈值判断是否删除自相关性较大的列
    不分label：
        求出自相关系数，剔除相关系数大于阈值的col
    分label:
        求出分label的自相关系数，剔除所有label都相关的col
    '''
    
    if columnslist is None:
        columnslist = list(data.columns)
        
    if label_col is not None and label_col in columnslist:
        columnslist.remove(label_col)
        
    print('----- 开始对数据进行自相关性分析 -----')    
    
    if label_col is None:
        ths_data = data[columnslist]
        autocorr_df,drop_col = _autocorr(ths_data,threshold = threshold,method = method)
        if isplot:
            Data_plot.plot_autocorr(ths_data,columnslist)

            
    else:
        
        ths_data = data[columnslist]        
        autocorr_df,drop_col = _autocorr(ths_data,threshold = threshold)
        if isplot:
            ths_data = pd.concat([ths_data,label_col],axis=1)
            Data_plot.plot_autocorr(ths_data,columnslist,label_col=label_col)

        for key,group in data.groupby(label_col):
            ths_res,drop_col_label = _autocorr(group,threshold = threshold)
            drop_col = drop_col & set(drop_col_label)
            print('Label:{},该因子剔除列数：{}列,总剔除列数：{}列'.format(key,len(drop_col_label) ,len(drop_col) ))
            
    print('自相关分析列数：原数据 {} 列，数据列：{} 列，自相关剔除：{} 列'.format(data.shape[1],data.shape[1]-len(drop_col),len(drop_col)))                
    if isdrop:
        data = data.drop(list(drop_col),axis =1)

    return data,autocorr_df,drop_col

def linear_corr_analysis(data,y = None,columnslist = None,label_col = None,
                      threshold = 0.8,mix_method ='mean',isdrop = True):
    '''
    相关性分析，默认前面为x,最后一列为y。
    分别求每一列x与y的 线性相关，秩相关，剔除异常相关 等系数，进行绝对值排序
    分label 分别求每个label下 各列与y的相关系数
    '''
    if y is not None:
        data = pd.concat([data,y],axis=1)
    if columnslist is None:
        columnslist = list(data.columns)
    else:
        columnslist.append(y.columns)
        
    if label_col is not None and label_col in columnslist:
        columnslist.remove(label_col)
    print('----- 线性相关分析  -----')
    if label_col is None:
        data = data[columnslist]
        #求相关性系数矩阵
        corr_df = data.corr(method = 'pearson').iloc[:-1,[-1]]
        corr_df = corr_df.rename(columns = {columnslist[-1]:'corr_' + str(columnslist[-1])})
        corr_df = corr_df.sort_values(corr_df.columns[0])

        plt = Data_plot.plot_bar_analysis(corr_df,corr_df.columns,threshold = [threshold,-threshold])
        plt.title('correlation with '+str(columnslist[-1]))
        plt.show()
        #如果有nan值，则用0填充
        abscorr = abs(corr_df.iloc[:,0]).fillna(0)
        drop_col = list(corr_df.loc[abscorr<threshold].index)

    else:
        corr_df_total = pd.DataFrame()
        for key,group in data.groupby(label_col):
            #分group求相关系数矩阵
            ths_data = group[columnslist]
            corr_df = ths_data.corr(method ='pearson').iloc[:-1,[-1]]
            corr_df = corr_df.rename(columns = {columnslist[-1]:'corr_'+ str(key) +'_' + columnslist[-1]})
            corr_df_total = pd.concat([corr_df_total,corr_df],axis=1)
        #group聚合    
        if mix_method == 'max': 
            mix_corr_df = pd.DataFrame(corr_df_total.max(axis = 1),columns =['max_label_corr_'+columnslist[-1]])
        elif mix_method == 'min':
            mix_corr_df = pd.DataFrame(corr_df_total.min(axis = 1),columns =['min_label_corr_'+columnslist[-1]])
        elif mix_method == 'mean':
            mix_corr_df = pd.DataFrame(corr_df_total.mean(axis = 1),columns =['mean_label_corr_'+columnslist[-1]])
        
        mix_corr_df = mix_corr_df.sort_values(mix_corr_df.columns[0])

        plt = Data_plot.plot_bar_analysis(mix_corr_df,mix_corr_df.columns,threshold = [threshold,-threshold])
        plt.title(mix_method + ' correlation with '+str(columnslist[-1]))
        plt.show()
        
        drop_col = list(mix_corr_df.loc[abs(mix_corr_df.iloc[:,0])<threshold].index)
        corr_df = mix_corr_df
    print('线性相关分析(阈值：{})结果：原数据：{}列，剔除数据{}列，筛选出：{}列。'.format(
            threshold,corr_df.shape[0],len(drop_col),corr_df.shape[0]-len(drop_col)))
    if isdrop:
        data = data.drop(list(drop_col),axis =1)
        if y is not None:
            data = data.drop(pd.DataFrame(y).columns[0],axis =1)
            data = data.reindex(index = y.index)
    return data,corr_df,drop_col

def nonlinear_corr_analysis(data,y = None,columnslist = None,label_col = None,threshold = 0.8,
                            method ='spearman',mix_method = 'max',isdrop = True):
    '''
    非线性相关系数
    秩相关
    距离相关
    '''
    if y is not None:
        data = pd.concat([data,y],axis=1)
    if columnslist is None:
        columnslist = list(data.columns)
    else:
        columnslist.append(y.columns)
    if label_col is not None and label_col in columnslist:
        columnslist.remove(label_col)
    print('-----  非线性相关分析  -----')
    if label_col is None:
        ths_data = data[columnslist]
        if method == 'spearman' or method == 'kendall':
            corr_df = ths_data.corr(method = method).iloc[:-1,[-1]]
            corr_df = corr_df.rename(columns = {columnslist[-1]:
                                        'corr_' + columnslist[-1]})
        elif method == 'distance':
            d_corr = []
            ths_data_y = np.array(data[columnslist[-1]])
            
            for col_x in columnslist[:-1]:
                try:
                    ths_data_x = np.array(data[col_x])
                    d_corr.append(dcor.dcor(ths_data_x,ths_data_y))
                except:
                    d_corr.append(np.nan)

            corr_df = pd.DataFrame(d_corr,index = columnslist[:-1],columns =['distance correlation with '+ str(columnslist[-1])])
        
        corr_df =corr_df.replace(np.nan,0)
        corr_df = corr_df.sort_values(corr_df.columns[0])
        
        plt = Data_plot.plot_bar_analysis(corr_df,corr_df.columns,threshold = [threshold,-threshold])
        plt.title('correlation with '+str(columnslist[-1]))
        plt.show()
        #如果有nan值，则用0填充
        abscorr = abs(corr_df.iloc[:,0]).fillna(0)
        drop_col = list(corr_df.loc[abscorr<threshold].index)

    else:
        label = set(data[label_col])
        corr_df_total = pd.DataFrame()
          
        for lab in label :     
            ths_data = data.loc[data[label_col] == lab]
            ths_data = ths_data[columnslist]
            
            if method == 'spearman' or method == 'kendall':
                corr_df = ths_data.corr(method = method).iloc[:-1,[-1]]
                corr_df = corr_df.rename(columns = {columnslist[-1]:
                                'corr_'+ str(lab) +'_' + columnslist[-1]})

            elif method == 'distance':
                d_corr = []
                ths_data_y = np.array(data[columnslist[-1]])
                for col_x in columnslist[:-1]:
                    ths_data_x = np.array(data[col_x])
                    d_corr.append(dcor.dcor(ths_data_x,ths_data_y))
                corr_df = pd.DataFrame(d_corr,index = columnslist[:-1],
                                   columns =['corr_'+ str(lab) +'_' + columnslist[-1]])
                
            corr_df_total = pd.concat([corr_df_total,corr_df],axis=1)
            

        if mix_method == 'max': 
            mix_corr_df = pd.DataFrame(corr_df_total.max(axis = 1),columns =['max_label_corr_'+columnslist[-1]])
        elif mix_method == 'min':
            mix_corr_df = pd.DataFrame(corr_df_total.min(axis = 1),columns =['min_label_corr_'+columnslist[-1]])
        elif mix_method == 'mean':
            mix_corr_df = pd.DataFrame(corr_df_total.mean(axis = 1),columns =['mean_label_corr_'+columnslist[-1]])
       
        mix_corr_df = mix_corr_df.sort_values(mix_corr_df.columns[0])
        corr_df = mix_corr_df
        
        plt = Data_plot.plot_bar_analysis(corr_df,corr_df.columns,threshold = [threshold,-threshold])
        plt.title(mix_method + ' correlation with '+str(columnslist[-1]))
        plt.show()
        #如果有nan值，则用0填充
        abscorr = abs(corr_df.iloc[:,0]).fillna(0)
        drop_col = list(corr_df.loc[abscorr<threshold].index)
#        drop_col = list(mix_corr_df.loc[abs(mix_corr_df.iloc[:,0])<threshold].index)
        
    print('非线性相关分析(阈值：{}结果)：原数据：{}列，剔除数据{}列，筛选出：{}列。'.format(
            threshold,corr_df.shape[0],len(drop_col),corr_df.shape[0]-len(drop_col)))            
    if isdrop:
        data = data.drop(list(drop_col),axis =1)
        if y is not None:
            data = data.drop(pd.DataFrame(y).columns[0],axis =1)
            data = data.reindex(index = y.index)
    return data,corr_df,drop_col

class G2G_analysis():
    '''
    组对组分析：
    1. KS 检验： 检验2组数据分布是否一致，True 分布不一致，False 分布一致
    2. 一列KS两两检验。
    3.全列两组 均值，标准差，四分卫距对比，获取变异最大的列。
    '''
    
    def __init__(self):
        pass
    
    def KS_test(self,group1,group2,p_threshold = 0.05):
        
        ks_value,p_value = ks_2samp(group1,group2)
        if p_value < p_threshold:
            return False
        else:
            return True

                
    def g2g_diff(self,df1,df2,method = 'mean'):
        '''
        分别比较两组均值差异,标准差差异，四分位数重合度
        
        '''
        if method == 'mean':
            df1_mean = df1.mean(axis = 0)
            df2_mean = df2.mean(axis = 0)
            res = pd.DataFrame(abs(df1_mean - df2_mean).sort_values(),columns = ['diff_mean']) 
        elif method == 'std':
            df1_std = df1.std(axis = 0)
            df2_std = df2.std(axis = 0)
            res = pd.DataFrame(abs(df1_std - df2_std).sort_values(),columns = ['diff_std']) 
        
        elif method == '4q':
            res = []
            for i in range(df1.shape[1]):
                df1_25 = np.percentile(df1.iloc[:,i],25)
                df1_75 = np.percentile(df1.iloc[:,i],75)
                
                df2_25 = np.percentile(df2.iloc[:,i],25)
                df2_75 = np.percentile(df2.iloc[:,i],75)
                
                if df2_25 < df1_75:
                    diff = df1_75 - df2_25 
                elif df1_25 < df2_75:
                    diff = df2_75 - df1_25 
                
                if diff < 0:
                    diff = 0
                
                res.append(diff)
            res = pd.DataFrame(res,index = df1.columns,columns = ['4q']).sort_values('4q')
            
        return res
    
    def get_KS_mat(self,data,label_col):
        
        if data.shape[1] > 2:
           
            if data.columns[0] != label_col:
                data = data.loc[:,[label_col]+[data.columns[0]]]
            else:
                data = data.loc[:,[label_col]+[data.columns[1]]]
        
        groupindex = data[label_col].drop_duplicates()
        res = pd.DataFrame(np.ones((len(groupindex),len(groupindex))),index = groupindex ,columns =groupindex)
        
        for key1,group1 in data.groupby(label_col):
            for key2,group2 in data.groupby(label_col):
                if key1 != key2:
                    KS_p = self.KS_test(group1.iloc[:,1].values,group2.iloc[:,1].values)
                else:
                    KS_p = True
                    
                res.loc[key1,key2] = KS_p
                
        return res
    
    def g2g_anaysis(self,data,label_col,columnslist = None ,method = 'mean'):
        '''
        限制两组进行对比
        '''
        groupindex = data[label_col].drop_duplicates()
        print(groupindex[0],'vs',groupindex[1],'method:',method)
        group1 = data[data[label_col] == groupindex[0]]
        group2 = data[data[label_col] == groupindex[1]]
        
        group1 = group1.drop(label_col,axis=1)
        group2 = group2.drop(label_col,axis=1)
        
        res = self.g2g_diff(group1,group2,method = method)
        
        #筛选最后5列
        col = res.index[-5:]
        Data_plot.plot_describe(data,label_col= label_col,columnslist=col)
        return res
    
class Outlier_analysis():
    '''
    异常点分析
    1.获取异常点分布矩阵
    2.使用IForest检测异常点
    3.确认xy异常点相关性
    '''
    def __init__(self,method = '3sigma', muti = 3,th):
        self.method = method
        self.muti = muti
        
    def get_outlier_mat(self,df):
        res = copy.copy(df)
        if self.method == '3sigma':
            self.upline = df.mean() + self.muti * df.std()
            self.dnline = df.mean() - self.muti * df.std()
        elif self.method == 'tukey':
            df = df.fillna(method = 'ffill')
            n25 = pd.DataFrame(np.percentile(df,25,axis=0),index = df.columns)
            n75 = pd.DataFrame(np.percentile(df,75,axis=0),index = df.columns)
            self.upline = n75 + self.muti * ( n75 - n25 )
            self.dnline = n25 - self.muti * ( n75 - n25 )

        res[df>self.upline] = 1
        res[df<self.dnline] = 1
        
        res[(df>=self.dnline)&(df<=self.upline)] = 0 
        
        return res
    
    def check_outlier_iforest(self,df,isplot = True):
        
        iforest = IsolationForest()
        iforest.fit(df)
        res = pd.DataFrame(iforest.predict(df),index = df.index,columns = ['outlier'])
        
        if isplot:
            #画数据分布散点图
            fr = Data_feature_reduction.Feature_Reduction(2)
            fr.fit(df)
            pca_res = fr.transform(df)
            if pca_res.shape[1] > 2:
                pca_res = pca_res.iloc[:,:2]
            
            plotdata = pd.concat([res,pca_res],axis = 1 )
            plt = Data_plot.plot_scatter(plotdata,label_col='outlier')
            plt.show()
            
        return res
    
    def xy_outlier_corr(self,x,y):
        
        x_outlier = self.get_outlier_mat(x)
        y_outlier = self.get_outlier_mat(y)

        data,corr_df,drop_col = nonlinear_corr_analysis(x_outlier,y_outlier)
        
        return corr_df
    
def ishave_Outlier(df):
    '''
    获取超过3sigma的列名
    '''
    upline = df.mean() + 3 * df.std()
    dnline = df.mean() - 3 * df.std()

    res = []
    check_mat = df[(df>upline)|(df<dnline)]
    
    for col in check_mat.columns:
        if len(check_mat.loc[:,col].dropna()):
            res.append(col)
    
    res = list(set(res))
    return res

def granger_causal_analysis(data,columnslist = None,label_col = None,threshold = 0.8,
                      mix_method = 'max',k=1,m=1,isdrop = False):
    '''
    格兰杰因果关系
    '''
    if columnslist is None:
        columnslist = list(data.columns)
        
    if label_col is None:
        g_c = []
        ths_data_y = np.array(data[columnslist[-1]])
        ths_data_y = ths_data_y.reshape([len(ths_data_y),1])
        for col_x in columnslist[:-1]:
            ths_data_x = np.array(data[col_x])
            ths_data_x = ths_data_x.reshape([len(ths_data_x),1])
            c= CausalCalculator.CausalCalculator(ths_data_x,ths_data_y)
            g_c.append(c.calcGrangerCausality(k,m))

        corr_df = pd.DataFrame(g_c,index = columnslist[:-1],columns =['granger_causal with '+ str(columnslist[-1])])
        corr_df = corr_df.sort_values(corr_df.columns[0])

        plt = Data_plot.plot_bar_analysis(corr_df,corr_df.columns,threshold = [threshold,-threshold])

        plt.ylim([-1.1,1.1])
        plt.title('granger_causal with '+str(columnslist[-1]))
        plt.show()
        drop_col = list(corr_df.loc[abs(corr_df.iloc[:,0])<threshold].index)

    else:
        label = set(data[label_col])
        corr_df_total = pd.DataFrame()
              
        for lab in label :
            ths_data = data.loc[data[label_col] == lab]
            ths_data = ths_data[columnslist]
            ths_data_y = np.array(ths_data[columnslist[-1]])
            ths_data_y = ths_data_y.reshape([len(ths_data_y),1])
            g_c = []
            for col_x in columnslist[:-1]:
                ths_data_x = np.array(ths_data[col_x])
                ths_data_x = ths_data_x.reshape([len(ths_data_x),1])
                c= CausalCalculator.CausalCalculator(ths_data_x,ths_data_y)
                g_c.append(c.calcGrangerCausality(k,m))


            corr_df = pd.DataFrame(g_c,index = columnslist[:-1],columns =['granger_causal '+ str(lab)+' '+ str(columnslist[-1])])
            corr_df_total = pd.concat([corr_df_total,corr_df],axis=1)
        
        plt = Data_plot.plot_bar_analysis(corr_df_total,corr_df_total.columns,threshold = [threshold,-threshold])
        plt.show()
        
        if mix_method == 'max': 
            mix_corr_df = pd.DataFrame(corr_df_total.max(axis = 1),columns =['max_label_corr_'+ columnslist[-1]])
        elif mix_method == 'min':
            mix_corr_df = pd.DataFrame(corr_df_total.min(axis = 1),columns =['max_label_corr_'+columnslist[-1]])
        elif mix_method == 'mean':
            mix_corr_df = pd.DataFrame(corr_df_total.mean(axis = 1),columns =['max_label_corr_'+columnslist[-1]])
        
        plt = Data_plot.plot_bar_analysis(mix_corr_df,mix_corr_df.columns,threshold = [threshold,-threshold])
        plt.ylim([-1.1,1.1])
        plt.title(mix_method + ' granger_causal '+str(columnslist[-1]))
        plt.show()
        drop_col = list(mix_corr_df.loc[abs(mix_corr_df.iloc[:,0])<threshold].index)
        
    if isdrop:
        data = data.drop(list(drop_col),axis =1)
        
    return data,corr_df,drop_col

def entropy_analysis(data,columnslist = None,label_col = None,threshold = 0.8,
                      mix_method = 'max',isdrop = False):
    '''
    计算各个列的熵
    '''
    
    if columnslist is None:
        columnslist = list(data.columns)
    if label_col is not None and label_col in columnslist:
        columnslist.remove(label_col)
    print('开始对数据进行熵值分析......')   
#    pdb.set_trace()
    if label_col is None:
        entropy_list = []
        for col_x in columnslist:
            if __check_discrete(data[col_x],check_num =20):
                ths_data_x = np.array(data[col_x])
#                ths_data_x = ths_data_x.reshape([len(ths_data_x),1])
                ent = discrete.entropy(ths_data_x)
            else:
                ths_data_x = np.array(data[col_x])
                ths_data_x = ths_data_x.reshape([len(ths_data_x),1])
                ent = continuous.entropy(ths_data_x,method ='gaussian')
            entropy_list.append(ent)

        df = pd.DataFrame(entropy_list,index = columnslist,columns =['entropy with '+ str(columnslist[-1])])
        df = df.sort_values(df.columns[0])
        plt = Data_plot.plot_bar_analysis(df,df.columns)
        plt.title('entropy')
        plt.show()
        
        drop_col = list(df.loc[abs(df.iloc[:,0])<threshold].index)
        
    else:
        label = set(data[label_col])
        df_total = pd.DataFrame()
              
        for lab in label :
            ths_data = data.loc[data[label_col] == lab]
            ths_data = ths_data[columnslist]
            entropy_list = []
            for col_x in columnslist:
                if __check_discrete(data[col_x],check_num =20):
                    try:
                        ths_data_x = np.array(data[col_x])
                        ent = discrete.entropy(ths_data_x)
                    except:
                        ent = np.nan
                else:
                    try:
                        ths_data_x = np.array(data[col_x])
                        ths_data_x = ths_data_x.reshape([len(ths_data_x),1])
                        ent = continuous.entropy(ths_data_x,method ='gaussian')
                    except:
                        ent = np.nan
                entropy_list.append(ent)

            df = pd.DataFrame(entropy_list,index = columnslist,columns =['entropy by '+ str(lab)+' '+ str(columnslist[-1])])
            df_total = pd.concat([df_total,df],axis=1)
        
        plt = Data_plot.plot_bar_analysis(df_total,df_total.columns)
        plt.show()
        
        if mix_method == 'max': 
            mix_corr_df = pd.DataFrame(df_total.max(axis = 1),columns =['max_label_corr_'+ columnslist[-1]])
        elif mix_method == 'min':
            mix_corr_df = pd.DataFrame(df_total.min(axis = 1),columns =['max_label_corr_'+columnslist[-1]])
        elif mix_method == 'mean':
            mix_corr_df = pd.DataFrame(df_total.mean(axis = 1),columns =['max_label_corr_'+columnslist[-1]])
        
        mix_corr_df = mix_corr_df.sort_values(mix_corr_df.columns[0])
        
        plt = Data_plot.plot_bar_analysis(mix_corr_df,mix_corr_df.columns)
        plt.title(mix_method + ' entropy')
        plt.show()
        
        drop_col = list(mix_corr_df.loc[abs(mix_corr_df.iloc[:,0])<threshold].index)
        
        
    if isdrop:
        data = data.drop(list(drop_col),axis =1)
        
    return data,df,drop_col

def search_machine(x,y,mapdict = None ,wrongcode = 0 ,method = 1):
    '''
    机台集中性计算方法：
    method 1: 
        使用熵值法定位异常机台
    method 2:
        使用关键因子法确认异常机台
    '''
    if method  == 1:
        #计算各个列的熵值
        idx = y[y==wrongcode].index
        ent_x = x.loc[idx,:]
        res_tot = pd.DataFrame()
        ent_x,entlist,col = entropy_analysis(ent_x)
        len_x = len(entlist[entlist == min(entlist.iloc[:,0])].dropna())
        if  len_x >5:
            chose = list(entlist.index)[:len_x]
        else:
            chose = list(entlist.index)[:5]
        
        #计算选定工序各机台不良率
        for step in chose:
            temp_data = pd.concat([x.loc[:,[step]],y],axis =1)
            for mechine in set(x.loc[:,step]):           
                pct = len(temp_data[(temp_data.loc[:,step]==mechine) & (temp_data.iloc[:,1] == wrongcode)]) /len(temp_data[temp_data.loc[:,step]==mechine])
                res = pd.DataFrame([step,mechine,pct],index=['step','mechine','pct']).T
                res_tot = pd.concat([res_tot,res])
        res_tot = res_tot.sort_values(['pct'],ascending =False)
        res_tot = res_tot[res_tot.iloc[:,2]>0]

        if mapdict is not None:
            for i in range(len(res_tot)):
                res_tot.iloc[i,1] = mapdict[res_tot.iloc[i,0]][res_tot.iloc[i,1]]
        plt = Data_plot.plot_bar_analysis(res_tot.set_index('mechine'),['pct'])
        plt.title(' NG rate')
        plt.show()
        return res_tot
    
def keyfeature_check(data,columnslist = None,label_col = None):
    '''
    关键因子检验：对比各类分布
    '''
    
    if columnslist is None:
        columnslist = data.columns
    #对数据进行标准化
    keyfeature = data.loc[:,columnslist]
    keyfeature,scaler = Data_Preprocess.data2avgstd(keyfeature)
    keyfeature = pd.DataFrame(keyfeature,columns = keyfeature.columns,index = keyfeature.index)
    if label_col is not None:
        keyfeature = pd.concat([keyfeature,data[[label_col]]],axis =1 )
        keyfeature.boxplot(column = columnslist,by='Y')
    else:
        keyfeature.boxplot(column = columnslist)
        
def reduce_dim_cluster(x,n_cluster = 2, dim = 2):
    '''
    将x 通过pca降成2维，或者3维，再通过聚类,画出效果图
    '''
    if dim ==2 or dim ==3:
        #标准化
#        x,scaler = Data_Preprocess.data2avgstd(x)
        #降维
        dr = Data_feature_reduction.Feature_Reduction(dim)
        dr.fit(x)
        x_dr = dr.transform(x).iloc[:,0:2]
        
        #聚类
        clu = KMeans(n_clusters=n_cluster).fit(x_dr)
        clu_label = clu.predict(np.array(x_dr))
#        pdb.set_trace()
        if dim ==2:
            x_dr = pd.DataFrame(x_dr.values,columns = ['x1','x2'],index = x_dr.index)
        else:
            x_dr = pd.DataFrame(x_dr.values,columns = ['x1','x2','x3'],index = x_dr.index)
        clu_label = pd.DataFrame(clu_label,columns = ['label'])
        plot_data = pd.concat([x_dr,clu_label],axis = 1)
        plt = Data_plot.plot_scatter(plot_data,label_col ='label')
        plt.show()
        res = pd.DataFrame(np.array(clu_label),columns =['clu_label'],index = x.index)
        return res
    else:
        return None
    
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    
    x1 = pd.DataFrame(np.random.normal(0,2,1000),columns = ['data'])
    label1 = pd.DataFrame(np.ones(x1.shape),columns = ['label'])
    x1 = pd.concat([label1,x1],axis=1)
    
    x2 = pd.DataFrame(np.random.normal(1,2,1000),columns = ['data'])
    label2 = pd.DataFrame(np.ones(x2.shape),columns = ['label'])
    x2 = pd.concat([label2,x2],axis=1)
    
    data = pd.concat([x1,x2])
    
    describe_analysis(data,columnslist=data.columns,label_col =  'label')
            
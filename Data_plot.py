# -*- coding: utf-8 -*-
"""
数据可视化
1.基础图表
1.1散点图 plot_scatter
1.2折线图 plot_line
1.3直方图 plot_hist
2.4箱线图 plot_box
    data: 数据，df.
    columnslist 筛选列 list
    label_col:标签列 str
2.数据分析图表
2.1 描述性统计 plot_describe
2.2 自相关性 plot_autocorr
"""
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pdb

c = ['red','blue','green','gold','y'] 

def plot_scatter(data,columnslist = None,label_col = None):
    '''
    一维散点：x轴：range(len(data)) ,y轴：data_x.
    二维散点：x轴：data[0],y轴：data[1] 
    三维散点：x轴：data[0],y轴：data[1] z轴：data[2],z轴用颜色表示
    
    '''
    global c 
    if columnslist is None:
        columnslist = list(data.columns)
    if label_col is not None and label_col in columnslist:
        columnslist.remove(label_col)
    
    len_col = len(columnslist)
        
    if len_col == 1:
        if label_col is None:
            y = np.array(data[columnslist]) 
            x = np.array(range(len(y)))
                
            plt.scatter(x,y)
            plt.xlabel('No.')
            plt.ylabel(columnslist[0])
            
        else:
            label = set(data[label_col])
            for i,lab in enumerate(label):
                ths_data = data[data[label_col] == lab]                
                y = np.array(ths_data[columnslist]) 
                x = np.array(ths_data.index)
                
                plt.scatter(x, y, c=c[i],marker='o', label=lab)
                
            plt.xlabel('No.')
            plt.ylabel(columnslist[0])
            plt.legend(loc='best')
            
    elif len_col == 2:
        if label_col is None:
            sns.jointplot(columnslist[0], columnslist[1], data=data ,kind="scatter")
            
        else:
            label = set(data[label_col])
            for i,lab in enumerate(label):
                ths_data = data[data[label_col] == lab]
                x = np.array(ths_data[columnslist[0]])                 
                y = np.array(ths_data[columnslist[1]]) 
                
                plt.scatter(x, y, c=c[i],marker='o', label=lab)
                
            plt.xlabel(columnslist[0])
            plt.ylabel(columnslist[1])
            plt.legend(loc='best')
    
    elif len_col >= 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if label_col is None:
            x = np.array(data[columnslist[0]]) 
            y = np.array(data[columnslist[1]])
            z = np.array(data[columnslist[2]])
            
            ax.scatter(x, y, z, marker='o')
            
        else:
            label = set(data[label_col])
            for i,lab in enumerate(label):
                ths_data = data[data[label_col] == lab]
                x = np.array(ths_data[columnslist[0]])                 
                y = np.array(ths_data[columnslist[1]]) 
                z = np.array(ths_data[columnslist[2]])
                
                ax.scatter(x, y, z, c=c[i],label =lab)
            
        ax.set_xlabel(columnslist[0])
        ax.set_ylabel(columnslist[1])
        ax.set_zlabel(columnslist[2])
        ax.legend(loc='best')
        
    return plt

def plot_line(data,columnslist = None,label_col = None, c = None):
    '''
    一维折线：x轴：range(len(data)) ,y轴：data.
    二维折线：x轴：data[0],y轴：data[1] 
    三维折线：x轴：data[0],y轴：data[1] z轴：data[2],z轴用颜色表示
    
    data: 
    '''
    if columnslist is None:
        columnslist = data.columns
    if label_col is not None and label_col in columnslist:
        columnslist.remove(label_col)
        
    if c is None:
        c = ['red','blue','green','gold','y'] 
        
    len_col = len(columnslist)
    
    if len_col == 1:
        if label_col is None:
            y = np.array(data[columnslist]) 
            x = np.array(range(len(y)))
                
            plt.plot(x,y,c[0])
            plt.xlabel('No.')
            plt.ylabel(columnslist[0])
            
        else:
            label = set(data[label_col])
            for i,lab in enumerate(label):
                ths_data = data.sort_values(label_col).reset_index()
                ths_data = ths_data[ths_data[label_col] == lab]                
                y = np.array(ths_data[columnslist]) 
                x = np.array(ths_data.index)
                
                plt.plot(x, y, c[i], label=lab)
                
            plt.xlabel('No.')
            plt.ylabel(columnslist[0])
            plt.legend(loc='best')
            
    elif len_col == 2:
        if label_col is None:
            x = np.array(data[columnslist[0]]) 
            y = np.array(data[columnslist[1]])
                
            plt.plot(x,y)
            plt.xlabel(columnslist[0])
            plt.ylabel(columnslist[1])
            
        else:
            label = set(data[label_col])
            for i,lab in enumerate(label):
                ths_data = data[data[label_col] == lab]
                x = np.array(ths_data[columnslist[0]])                 
                y = np.array(ths_data[columnslist[1]]) 
                
                plt.plot(x, y, c=c[i], label=lab)
                
            plt.xlabel(columnslist[0])
            plt.ylabel(columnslist[1])
            plt.legend(loc='best')
            
    elif len_col == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if label_col is None:
            x = np.array(data[columnslist[0]]) 
            y = np.array(data[columnslist[1]])
            z = np.array(data[columnslist[2]])
            
            ax.plot(x, y, z)
            
        else:
            label = set(data[label_col])
            for i,lab in enumerate(label):
                ths_data = data[data[label_col] == lab]
                x = np.array(ths_data[columnslist[0]])                 
                y = np.array(ths_data[columnslist[1]]) 
                z = np.array(ths_data[columnslist[2]]) 
                ax.plot(x, y, z, c=c[i],label =lab)
                
        ax.set_xlabel(columnslist[0])
        ax.set_ylabel(columnslist[1])
        ax.set_zlabel(columnslist[2])
        ax.legend(loc='best')
        
    return plt

def plot_hist(data,columnslist,label_col = None,bins =10):
    '''
    一维直方图：
    '''
    c = ['red','blue','green'] 
    
    if label_col is None:
        mu = data[columnslist[0]].mean()
        sigma = data[columnslist[0]].std()
        x = np.array(data[columnslist[0]])
        
        fig, ax1 = plt.subplots()
        n, bins, patches = ax1.hist(x, bins = bins,alpha = 0.7)
        ax2 = ax1.twinx()
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * 
             (1 / sigma * (bins - mu))**2))
        ax2.plot(bins, y, 'r--')
        ax2.set_xlabel('No.')
        ax2.set_y ([0, max(y)+0.1 ])
        ax1.set_ylabel(columnslist[0])
        
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        label = set(data[label_col])
        for i,lab in enumerate(label):
            ths_data = data[data[label_col] == lab]
            x = np.array(ths_data[columnslist[0]])
            hist, xedges = np.histogram(x, bins= bins)         
            ax.bar(xedges[1:], hist, zs=i, zdir='y', color=c[i], alpha=0.5,width = (x.max()-x.min())/bins)

        ax.set_ylabel(columnslist[0])
        ax.set_xlabel('No.')
        
    return plt

def plot_bar(data,columnslist,label_col = None,c = None):
    
    if c is None:
        c = ['red','blue','green'] 
        
    fig, ax = plt.subplots()

    if label_col is None:
        width = 0.85
        for i in range(len(columnslist)):
        
            index = data.index
            x = data[columnslist[i]]
            rects = ax.bar(np.arange(len(x))-width/2+i*width/len(columnslist), x,width/len(columnslist),
                    alpha=0.8, color=c[i],
                    label=columnslist[i])
            
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(data.index)
        ax.set_xlabel('Group')
        ax.set_ylabel(columnslist[i])
        ax.legend(loc=2)
        
    else:
        label = set(data[label_col])
        for i,lab in enumerate(label):
            ths_data = data.loc[data[label_col] == lab]
            x = np.array(ths_data[columnslist[0]])
            index = ths_data[columnslist[0]].index
            rects = ax.bar(index, x,
                    alpha=0.8, color=c[i],
                    label=lab)
        ax.set_xlabel('Group')
        ax.set_ylabel('Scores')
        ax.legend(loc='best')
        
    return plt

def plot_barh(x):
    idx = range(len(x))
    plt.barh(idx,np.array(x))
    plt.grid(axis='x')
    plt.yticks(idx,x.index)
    return plt
    

def plot_box(data,columnslist,label_col = None):
    if label_col is None:
        fig, ax = plt.subplots()
        ax.boxplot(data[columnslist[0]])
        
    else:
        fig, ax = plt.subplots()
        label = set(data[label_col])
        total_df = pd.DataFrame()
        for i,lab in enumerate(label):
            ths_data = data[data[label_col] == lab]
            total_df = pd.concat([total_df,ths_data[columnslist[0]]],axis=1)
        pdb.set_trace()
        total_df.boxplot()

    ax.set_xlabel(columnslist[0])

    return plt

def plot_spc(data,p1 = None , p2 = None , method ='3sigma'):
    '''
    spc管控图，只能画一维数据,输入的data，1维df.
    method = 3sigma p1 = mean , p2 = std
    如果传入mean,std 则使用传入的数据，否则使用输入的数据计算mean,std
    method = tukey
    '''

    #如果超过1维，则取第一列
    if data.shape[1] >=2:
        data = data.iloc[:,0]
        
    if method == '3sigma':
        if p1 is None:    
            p1 = data.mean().values[0]
        if p2 is None:
            p2 = data.std().values[0]
    
        std_2 = np.ones(data.shape) * (p1 + 2*p2)
        std_3 = np.ones(data.shape) * (p1 + 3*p2)
        _2_std = np.ones(data.shape) * (p1 - 2*p2)
        _3_std = np.ones(data.shape) * (p1 - 3*p2)
    
        x = np.array(range(len(data)))
        #数据走势线
        plt.plot(x, np.array(data), 'black',label = 'x')
        #管控线
        plt.plot(x, std_3, 'r--',label = '3sigma')
        plt.plot(x, _3_std, 'r--',label = '-3sigma')
        plt.plot(x, std_2, 'y--',label = '2sigma')
        plt.plot(x, _2_std, 'y--',label = '-2sigma')
        plt.title( '{} control chart : {}'.format(method,data.columns[0]))
        plt.legend(loc='upper left')
    if method == 'tukey':
        if p1 is None:    
            p1 = 1.5 
        if p2 is None:
            p2 = 3
    
        base_line1 =np.ones(data.shape) * (np.percentile(data,75)  + p1 * (np.percentile(data,75) - np.percentile(data,25)))
        base_line2 =np.ones(data.shape) * (np.percentile(data,75)  + p2 * (np.percentile(data,75) - np.percentile(data,25)))
        base_line_1 =np.ones(data.shape) * (np.percentile(data,25)  - p1 * (np.percentile(data,75) - np.percentile(data,25)))
        base_line_2 =np.ones(data.shape) * (np.percentile(data,25)  - p2 * (np.percentile(data,75) - np.percentile(data,25)))
    
        x = np.array(range(len(data)))
        #数据走势线
        plt.plot(x, np.array(data), 'black',label = 'x')
        #管控线
        plt.plot(x, base_line2, 'r--',label = '3R')
        plt.plot(x, base_line_2, 'r--',label = '-3R')
        plt.plot(x, base_line1, 'y--',label = '1.5R')
        plt.plot(x, base_line_1, 'y--',label = '-1.5R')
        plt.title('{} control chart : {}'.format(method,data.columns[0]))
        plt.legend(loc='upper left')
    return plt

def plot_describe(data,columnslist = None,label_col = None):
    '''
    画分布分析图
    '''
    
    if columnslist is None:
        columnslist = list(data.columns)
        
    if label_col is not None and label_col in columnslist:
        columnslist.remove(label_col)
    #列太多画不下
    n_col = len(columnslist)
    if n_col> 5:
        columnslist = columnslist[0:5]
  
    if label_col is None:
        
        (f, axs) = plt.subplots(2, len(columnslist), figsize=(10, 8), sharex=True)
        for i in range(len(columnslist)):
            #画箱线图
            if len(columnslist) == 1:
                sns.boxplot(x=columnslist[i],data = data, ax=axs[0])
                axs[0].set_xlabel(columnslist[i])
            else:
                sns.boxplot(x=columnslist[i],data = data, ax=axs[0,i])
                axs[0,i].set_xlabel(columnslist[i])

            #画直方概率图
            if len(columnslist) == 1:
                sns.distplot(data.loc[:,columnslist[i]].dropna(),ax=axs[1])
                axs[1].set_xlabel(columnslist[i])
            else:
                sns.distplot(data.loc[:,columnslist[i]].dropna(),ax=axs[1,i])
                axs[1,i].set_xlabel(columnslist[i])

    else:
        (f, axs) = plt.subplots(2, len(columnslist), figsize=(10, 8), sharex=True)
        for i in range(len(columnslist)):
            #画箱线图
            sns.boxplot(x=columnslist[i],y=label_col,data = data,ax=axs[0,i])
            axs[0,i].set_xlabel(columnslist[i])
            #画直方概率图
            for key,group in data.groupby(label_col):
                sns.kdeplot(group.loc[:,columnslist[i]].dropna(), shade =True,label=key,ax=axs[1,i])
            axs[1,i].set_xlabel(columnslist[i])
                     
    return plt

def plot_autocorr(data,columnslist = None,label_col = None):
    '''
    画自相关图
    scale:画图的尺寸
    '''
    if columnslist is None:
        columnslist = list(data.columns)
    #列太多画不下
    if len(columnslist)> 15:
        columnslist = columnslist[0:15]
        
    data = data.loc[:,columnslist].dropna()
    if label_col is None:
        plt = sns.pairplot(data)
    else:
        plt = sns.pairplot(data, hue = label_col)
    return plt

def plot_bar_analysis(data,columnslist = None ,label_col = None,threshold = None):
    '''
    实现画有阈值的bar分析图。
    输入的数据：
    data corr_label1 corr_label2 
    col1  0.5        0.7
    col2  -0.3       0.2
    col3  -0.7       0.5
    
    '''
    data = data.dropna()
    if columnslist is None:
        columnslist = data.columns
        
    for col in columnslist:
        if data.shape[0] > 30:
            data = pd.concat([data.iloc[0:15,:],data.iloc[-15:,:]])
            
        if data.shape[0] > 5:
            #超过15个数据，竖向排列
            ndarry_x = np.array(data.loc[:,col])
            ndarry_y = np.array(data.index)
        else:
            #不超过15个 横向排列
            ndarry_x = np.array(data.index)
            ndarry_y = np.array(data.loc[:,col])
        # 画bar图    
        sns.barplot(x=ndarry_x, y=ndarry_y, palette="Blues_d")
        # 画阈值
        if threshold is not None:
            for thres in threshold: 
                if data.shape[0] > 5:
                    line_data_y = np.array(range(data.shape[0]))
                    line_data_x = np.ones(data.shape) * thres
                else:
                    line_data_x = np.array(range(data.shape[0]))
                    line_data_y = np.ones((data.shape[0],1)) * thres
                plt.plot(line_data_x,line_data_y,'r--')

    return plt

def plot_confusion_matrix(y_truth, y_predict, cmap=plt.cm.Blues):

    cm = confusion_matrix(y_truth, y_predict)
    plt.matshow(cm, cmap=cmap)  
    plt.colorbar()  

    for x in range(len(cm)): 
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.xlabel('True label') 
    plt.ylabel('Predicted label')  
    return plt


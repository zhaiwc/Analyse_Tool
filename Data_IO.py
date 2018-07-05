# -*- coding: utf-8 -*-
"""
文件输入输出
输入：xls,csv,txt.
输出：xls,csv,txt.
"""


import pandas as pd
import pickle

def get_txt_data(file_path,sep = None):
    try:
        print('reading data ...')
        data = pd.read_table(file_path,sep = sep)
        print('reading success ...')
    except FileNotFoundError as e:
        data = None
    finally:
        return data

def get_csv_data(file_path):
    try:
        print('reading data ...')
        data = pd.read_csv(file_path)
        print('reading success ...')
    except:
        data = None
    finally:
        return data
    
def get_xls_data(file_path):
    try:
        print('reading data ...')
        data = pd.read_excel(file_path)
        print('reading success ...')
    except:
        data = None
    finally:
        return data
    
def save_data(data_name,data):
    if data is not None:
        print('saving data ...')
        with open(data_name,'wb') as file:
            data.to_pickle(file)
        print('saving success...')
    else:
        print('data is None ,please check')
        
def get_data(data_name):
    try:
        print('loading data ...')
        with open(data_name,'rb') as file:
            data = pickle.load(file)
        print('loading success...')
    except:
        print('data is None ,please check')
        data = None
    finally:   
        return data
        
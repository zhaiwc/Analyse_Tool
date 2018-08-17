# -*- coding: utf-8 -*-
"""
文件输入输出
输入：xls,csv,txt.
输出：xls,csv.
json输入输出
图片输入输出
"""


import pandas as pd
import json
import cv2
import pdb

def get_data(file_path,**kw):
    file_type = file_path.split('.')[1]
    print('----- loading data -----')
    try:
        print('reading data ...')
        if file_type == 'txt':
            data = pd.read_table(file_path,**kw)
        elif file_type == 'csv':
            data = pd.read_csv(file_path,**kw)
        elif file_type == 'xls' or file_type == 'xlsx':
            data = pd.read_excel(file_path,**kw)
        print('reading success ...')
        
    except Exception as e:
        print('reading false,{}'.format(e))
        data = None
    finally:
        return data
    
def to_file(data,file_name,file_path = None, file_type = 'xls'):
    if file_path is not None:
        file_name = file_path +'\\'+ file_name
    print('saving data ... path: {}'.format(file_name))
    if file_type =='xls':
        data.to_excel(file_name)
    elif file_type =='csv':
        data.to_csv(file_name)
    print('saving success ...')

def json2py(json_str):
    print(json_str)
    json_dict = json.loads(json_str,encoding='utf-8')
    return json_dict

def py2json(obj):
    
    if type(obj) == pd.DataFrame or type(obj) == pd.Series:
        json_str = obj.to_json()
    else:
        for key in obj.keys():
            if type(obj[key]) == pd.DataFrame or type(obj[key]) == pd.Series:
                jsonstr = obj[key].to_json()
                json_dict = json.loads(jsonstr)
                obj[key] = json_dict
        json_str = json.dumps(obj)
        
    return json_str

def get_img(img_path):
    
    img = cv2.imread(img_path,0)
    return img

def save_img(img,file_name,file_path):
    try:
        if file_path is not None:
            file_name = file_path +'\\'+ file_name
        
        cv2.imwrite(file_path,img)
        return True
    except Exception as e:
        print('saving false,{}'.format(e))
        return False
    
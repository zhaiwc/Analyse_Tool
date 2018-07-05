# -*- coding: utf-8 -*-
"""
    功能：
    1.数据字典：
        project:
            data:
                (pd.DataFrame)
                data_name:
                data_path:
                remark:
            model:
                (pd.DataFrame)
                model_name:
                model_path:
                remark:
    
    2.保存路径：
        //project//data//data_name.dat
        //project//model//model_name.dat
    
    3.日志：
        path
    
    4.操作方法：
        数据：增删查
        模型：增删查
"""

import os
import shelve
import math
import pdb
import pandas as pd
import time


class LocalDB():

    def __init__(self,project_name):
        self.project_name = project_name
        
        self.root_path = 'D:\\python\\local_db'
        self.log_path = self.root_path +'\\log'
    def __open_dict(self):
        '''
        到指定路径下打开数据字典,返回该project下的字典,一定会新建空字典
        '''
        
        if os.path.exists(self.root_path):
            dict_path = self.root_path +'\\data_dict.dat'
            db = shelve.open(dict_path, 'c')
#            pdb.set_trace()
            data_dict = db['data_dict']
            db.close()
            if self.project_name in data_dict.keys():
                res = data_dict[self.project_name]
            else:
                res = {}
            
        else:
            os.mkdir(self.root_path)
            dict_path = self.root_path +'\\data_dict.dat'
            db = shelve.open(dict_path, 'c')
            data_dict = {}
            data_dict[self.project_name] = {}

            db['data_dict'] = data_dict
            res = data_dict[self.project_name]
            db.close()
            
        return res
    
    def __update_dict(self,new_dict):
        '''
        根据新输入的dict进行update
        '''
        dict_path = self.root_path +'\\data_dict.dat'
        db = shelve.open(dict_path, 'c')
        orgin_dict = db['data_dict']
        orgin_dict[self.project_name] = new_dict
        db['data_dict'] = orgin_dict
        db.close()
        
    def __open_log(self):
        '''
        到指定路径下打开数据字典,返回该project下的字典,一定会新建空字典
        '''
        
        if os.path.exists(self.log_path):
            log_path = self.log_path +'\\log.dat'
            db = shelve.open(log_path, 'c')
            data_log = db['log']
            db.close()
            if self.project_name in data_log.keys():
                res = data_log[self.project_name]
            else:
                res = {}
            
        else:
            os.mkdir(self.log_path)
            log_path = self.log_path +'\\log.dat'
            db = shelve.open(log_path, 'c')
            data_log = {}
            data_log[self.project_name] = {}

            db['log'] = data_log
            res = data_log[self.project_name]
            db.close()
            
        return res
    
    def __update_log(self,new_log):
        '''
        根据新输入的dict进行update
        '''
        log_path = self.log_path +'\\log.dat'
        db = shelve.open(log_path, 'c')
        orgin_log = db['log']
        orgin_log[self.project_name] = new_log
        db['log'] = orgin_log
        db.close()
        
    def __insert_dict_data(self,data_name,remark = '-'):
        '''
        在数据字典中插入数据信息，如果重复命名，则插入失败。
        '''
        data_dict = self.__open_dict()
        #创建文件路径
        if os.path.exists(self.root_path+ '\\' + str(self.project_name)) == False:
            os.mkdir(self.root_path+ '\\' + str(self.project_name))
        if os.path.exists(self.root_path+ '\\' + str(self.project_name + '\\data')) == False:
            os.mkdir(self.root_path+ '\\' + str(self.project_name + '\\data'))

        if 'data' in data_dict.keys():
            if data_name in data_dict['data']['data_name'].tolist():
                print('-----重复命名data_name:{}-----'.format(data_name))
                return False
            else:
                data_path = self.root_path + '\\' + str(self.project_name) + '\\data\\' + str(data_name)
                new_data = pd.DataFrame([data_name,remark,data_path],index = ['data_name','remark','data_path']).T
                new_data = pd.concat([data_dict['data'],new_data])
                data_dict['data'] = new_data
                self.__update_dict(data_dict)    
                return True
        else:
            data_path = self.root_path + '\\' + str(self.project_name) + '\\data\\' + str(data_name)
            data_dict['data'] = pd.DataFrame([data_name,remark,data_path],index = ['data_name','remark','data_path']).T
            self.__update_dict(data_dict)
            return True
    
    def __insert_dict_model(self,model_name,remark = '-'):
        '''
        在数据字典中插入模型信息，如果重复命名，则插入失败。
        '''
        data_dict = self.__open_dict()
        #创建文件路径
        if os.path.exists(self.root_path+ '\\' + str(self.project_name)) == False:
            os.mkdir(self.root_path+ '\\' + str(self.project_name))
        if os.path.exists(self.root_path+ '\\' + str(self.project_name + '\\model')) == False:
            os.mkdir(self.root_path+ '\\' + str(self.project_name + '\\model'))

        if 'model' in data_dict.keys():
            if model_name in data_dict['model']['model_name'].tolist():
                print('-----重复命名model_name:{}-----'.format(model_name))
                return False
            else:
                model_path = self.root_path + '\\' + str(self.project_name) + '\\model\\' + str(model_name)
                new_model = pd.DataFrame([model_name,remark,model_path],index = ['model_name','remark','model_path']).T
                new_model = pd.concat([data_dict['model'],new_model])
                data_dict['model'] = new_model
                self.__update_dict(data_dict)    
                return True
        else:
            model_path = self.root_path + '\\' + str(self.project_name) + '\\model\\' + str(model_name) 
            data_dict['model'] = pd.DataFrame([model_name,remark,model_path],index = ['model_name','remark','model_path']).T
            self.__update_dict(data_dict)
            return True
        
    def __del_dict_data(self,data_name):
        '''
        在数据字典中删除数据信息，如果数据字典中没有该数据，则删除失败。
        '''
        data_dict = self.__open_dict()
        temp = data_dict['data']
        temp = temp.set_index('data_name')
        if data_name in temp.index:
            temp = temp.drop(data_name)
            data_dict['data'] = temp.reset_index()
            self.__update_dict(data_dict)
            return True
        else:
            return False
        
    def __del_dict_model(self,model_name):
        '''
        在数据字典中删除模型信息，如果数据字典中没有该模型，则删除失败。
        '''
        data_dict = self.__open_dict()
        temp = data_dict['model']
        temp = temp.set_index('model_name')
        if model_name in temp.index:
            temp = temp.drop(model_name)
            data_dict['model'] = temp.reset_index()
            self.__update_dict(data_dict)
            return True
        else:
            return False
        
    def __get_data_by_path(self,data_path):
        '''
        根据路径获取data
        '''
        if os.path.exists(data_path + '.dat'):
            db = shelve.open(data_path, 'c')
            data = db['data']
            db.close()
        else:
            print('----- 数据路径不正确 -----')  
            data = None
        return data
    
    def __save_data_by_path(self,data_path,data):
        '''
        根据路径保存data
        '''
        db = shelve.open(data_path, 'c')
        db['data']=data
        db.close()
        
    def __del_data_by_path(self,data_path):
        '''
        根据路径删除data
        '''
        os.remove(data_path+'.dat')
        os.remove(data_path+'.dir')
        os.remove(data_path+'.bak')
#        db = shelve.open(data_path, 'c')
#        db.pop('data')
#        db.close()
        
    def __get_model_by_path(self,model_path):
        '''
        根据路径获取model
        '''
        if os.path.exists(model_path + '.dat'):
            db = shelve.open(model_path, 'c')
            model = db['model']
            db.close()
        else:
            print('----- 数据路径不正确 -----')  
            model = None
        return model
    
    def __save_model_by_path(self,model_path,model):
        '''
        根据路径保存model
        '''
        db = shelve.open(model_path, 'c')
        db['model']=model
        db.close()
        
    def __del_model_by_path(self,model_path):
        '''
        根据路径删除model
        '''
        os.remove(model_path+'.dat')
        os.remove(model_path+'.dir')
        os.remove(model_path+'.bak')
#        db = shelve.open(model_path, 'c')
#        db.pop('model')
#        db.close()
    
    def select_dict(self):
        '''
        查project名下所有的数据和模型
        '''
        data_dict = self.__open_dict()

#        if self.project_name in data_dict.keys():
#        print('----- project：{} -----'.format(self.project_name))
#        if 'data' in data_dict.keys():
#            print('----- data -----')
#            print(data_dict['data'])
#        if 'model' in data_dict.keys():
#            print('----- model -----')
##            print(data_dict['model'])
        if 'data' not in data_dict.keys() and 'model' not in data_dict.keys():
            print('-----没有该 project:{} 的信息 -----'.format(self.project_name))
#            print(data_dict)
        return data_dict
        
    def logged(func):
        # 日志装饰器
        def writelog(self,name,*args,**kwargs):
            dic={}
            dic['project'] = self.project_name
            dic['operate']=func.__name__
            dic['name']=name
            dic['res']=func(self,name,*args,**kwargs)
            dic['time']=time.strftime('%Y/%m/%d %H:%M:%S')
            new_log=pd.DataFrame(dic,index=[0],columns=['project','operate','name','res','time'])
            old_log = self.__open_log()
            if len(old_log):
                new_log = pd.concat([old_log,new_log])
            else:
                pass
            self.__update_log(new_log)
        return writelog

    def get_data(self,data_name):
        '''
        查project名下dataname对应的数据
        '''
        data_dict = self.__open_dict()
        if 'data' in data_dict.keys():
            data_info = data_dict['data']
            data_path = data_info[data_info.data_name == data_name]['data_path']
            if len(data_path):
                data_path = data_path.values[0]
                data = self.__get_data_by_path(data_path)
                return data
            else:
                print('----- 没有该 data_name:{}-----'.format(data_name))
                return False
        else:
            return False
    @logged
    def save_data(self,data_name,data,remark = '-'):
        '''
        将数据保存入指定的位置，并同时更新数据字典
        '''
        if self.__insert_dict_data(data_name,remark):
            data_dict = self.__open_dict()
            data_info = data_dict['data']
            data_path = data_info[data_info.data_name == data_name]['data_path'].values[0]
            self.__save_data_by_path(data_path,data)
            print('----- 保存数据成功，data_name:{}-----'.format(data_name))
            return True

        else:
            print('----- 保存数据失败，data_name:{}-----'.format(data_name))
            return False
    @logged
    def del_data(self,data_name):
        data_dict = self.__open_dict()
        if 'data' in data_dict.keys():
            data_info = data_dict['data']
            data_path = data_info[data_info.data_name == data_name]['data_path']
            if len(data_path):
                data_path = data_path.values[0]
                self.__del_data_by_path(data_path)
                self.__del_dict_data(data_name)
                print('----- 删除数据成功，data_name:{}-----'.format(data_name))
                return True
            else:
                print('----- 删除数据失败，data_name:{}-----'.format(data_name))
                return False
        else:
            return False

    def get_model(self,model_name):
        '''
        查project名下dataname对应的数据
        '''
        data_dict = self.__open_dict()
        
        if 'model' in data_dict.keys():
            model_info = data_dict['model']
            model_path = model_info[model_info.model_name == model_name]['model_path']
            if len(model_path):
                model_path = model_path.values[0]
                model = self.__get_model_by_path(model_path)
                return model
            else:
                print('----- 没有该 model_name:{}-----'.format(model_name))
                return False
        else:
            return False
    @logged
    def save_model(self,model_name,model,remark = '-'):
        '''
        将模型保存入指定的位置，并同时更新数据字典
        '''
        if self.__insert_dict_model(model_name,remark):
            data_dict = self.__open_dict()
#            pdb.set_trace()
            model_info = data_dict['model']
            model_path = model_info[model_info.model_name == model_name]['model_path'].values[0]
            self.__save_model_by_path(model_path,model)
            print('----- 保存模型成功，model_name:{}-----'.format(model_name))
            return True

        else:
            print('----- 保存模型失败，model_name:{}-----'.format(model_name))
            return False   
    @logged
    def del_model(self,model_name):
        data_dict = self.__open_dict()
        if 'model' in data_dict.keys():
            model_info = data_dict['model']
            model_path = model_info[model_info.model_name == model_name]['model_path']
            if len(model_path):
                model_path = model_path.values[0]
                self.__del_model_by_path(model_path)
                self.__del_dict_model(model_name)
                print('----- 删除模型成功 model_name:{}-----'.format(model_name))
                return True
            else:
                print('----- 删除模型失败 model_name:{}-----'.format(model_name))
                return False
        else:
            return False   
    
    def get_log(self):
        '''
        读取该project下的日志文件
        '''
        old_log = self.__open_log()
        if len(old_log):
            old_log = old_log[old_log.project == self.project_name]
            print(old_log)
        else:
            print('----- 没有log文件 project_name:{}-----'.format(self.project_name))

if __name__ == '__main__':
    test = LocalDB('test')
    data = pd.DataFrame([1,2,3,4])
#    pdb.set_trace()
#    test.del_model('data2')
#    test.save_model('data2',data,remark ='-' )
    test.get_log()
#    testdict = test.select_dict()
#    print(test.insert_dict_data('data3',remark='test_1'))
#    obj=LocalDB('obj')
#    pdb.set_trace()
#    print(obj.del_model('tesla'))
            
                
    
        
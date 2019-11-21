# -*- coding: utf-8 -*-

import numpy as np
import config as cfg
import time
import pickle

class data_progress(object):
    
    def __init__(object):
        pass
    
    def save_vid2label(self,root):
        cn = 0
        vid2label = {}
        ## vid-->label [itemid-->label]
        with open(root + '/ml_Item.csv') as f:
            for line in f.readlines():
                vid = line.strip()
                vid2label[vid] = cn
                cn = cn+1
        with open(root+"ml_Item.pkl","wb") as f:
            pickle.dump(vid2label, f)
            
    def save_dataInfo(self,path):
        
        userInf = []
        with open(path) as f:
            for line in f.readlines():
                seg = line.split(',')[1]
                data_time_list = seg.split("||")
                data_list = []
                time_list = []
                for item_stime in data_time_list:
                    item = item_stime.split(":")[0]
                    stime = int(item_stime.split(":")[1])
                    data_list.append(item)
                    time_list.append(stime)
                if len(data_list)<10:
                    continue
                for i in range(len(data_list)-9):
                    tmp = {}
                    tmp['history'] = data_list[i:i+9]
                    tmp['obj'] = data_list[i+9]
                    tmp['ex_age'] = time_list[i+9]
                    userInf.append(tmp)
                
        with open(path+".pkl","wb") as f:
            pickle.dump(userInf, f)
        
       
        

class ucf_data(object):
    def __init__(self,phase):

        if phase == 'train':
            self.batch_size = cfg.train_batch_size
            file_path = '/Users/liuhongbing/Documents/data/recommend/ml-20m/ml_userInfo/train.csv.pkl'
        else:
            self.batch_size = cfg.test_batch_size
            file_path = '/Users/liuhongbing/Documents/data/recommend/ml-20m/ml_userInfo/test.csv.pkl' 
        self.cursor = 0
        self.userInf = []
        self.vid2label = {}

        self.maxtime = 1427784002
        self.dtime = self.maxtime - 789652004

        with open("/Users/liuhongbing/Documents/data/recommend/ml-20m/ml_Items/ml_Itemsml_Item.pkl","rb") as f:
            self.vid2label = pickle.load(f)
        self.num_classes = len(self.vid2label)

        with open(file_path,"rb") as f:
            self.userInf = pickle.load(f)
          
        np.random.shuffle(self.userInf)  
        print('uderInf',len(self.userInf)) 
       

    def time_cal(self):
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(timeArray)
        return int(timestamp)

    def get(self):
        history = np.zeros((self.batch_size,9),np.int32)
        ex_age = np.zeros((self.batch_size,1),np.float32)
        labels = np.zeros((self.batch_size,1),np.int32)
        count = 0
        time_stamp = self.time_cal()
        while count < self.batch_size:
            data = self.userInf[self.cursor]
            history_vidlist = data['history']
            his_label_list = []
            for i in range(len(history_vidlist)):
                his_label_list.append(self.vid2label[history_vidlist[i]])
            history_label = np.array(his_label_list)
            tmp_label = self.vid2label[data['obj']]
            history[count, :] = history_label
            labels[count, :] = tmp_label
            ex_age[count, :] = (self.maxtime - data['ex_age'])/86400            
            #import pdb
            #pdb.set_trace()
            count = count+1
            self.cursor = self.cursor + 1
            
            if self.cursor >= len(self.userInf):
                np.random.shuffle(self.userInf)
                self.cursor = 0

        return history, ex_age, labels


def main():
    data = data_progress()
    data.save_vid2label("/Users/liuhongbing/Documents/data/recommend/ml-20m/ml_Items")
    data.save_dataInfo("/Users/liuhongbing/Documents/data/recommend/ml-20m/ml_userInfo/train.csv")
    data.save_dataInfo("/Users/liuhongbing/Documents/data/recommend/ml-20m/ml_userInfo/test.csv")
    
def main2():

    data = ucf_data('train')
    for i in range(20):
        start = time.time()
        history, ex_age, labels = data.get()
        t = time.time() - start
        print('history',history,history.shape)
        print('labels',labels,labels.shape)
        print('ex_age',ex_age, ex_age.shape)

if __name__ == '__main__':
    main2()
            
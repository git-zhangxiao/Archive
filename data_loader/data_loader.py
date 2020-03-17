from base.base_data_loader import BaseDataLoader
import os
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd


class ModelDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ModelDataLoader, self).__init__(config)
        self.config = config        
        #self.load_data()

    def load_data(self):
        #print("123")
        path = self.config['data_loader']['data_path']
        if self.config['data_loader']['name'] == 'TimeSeries':
            curt_index = 1
            prev_index = 0
            
            print('self.data_path',path)
            file_list = self.files_path_dir(path)
            print("file_list",file_list)
            for file in file_list:
                
                if curt_index <= len(file_list) - 1:
                    print("Loading tw-1 file:{} \n Loading tw files: {}".format(file_list[prev_index],file_list[curt_index]))
                    try:
                        df_tw_prev = self.load_from_file(file_list[curt_index])
                        df_tw_curt = self.load_from_file(file_list[prev_index])
                    except StopIteration:
                        break
                    #df.drop(labels=['增换购标志'], axis=1, inplace=True)
                    self.y_train = df_tw_prev[self.config['data_loader']['y_column']]
                    df_tw_prev.drop(labels=[self.config['data_loader']['y_column']], axis=1, inplace=True)
                    self.X_train = df_tw_prev.copy()
                    self.y_test = df_tw_curt[self.config['data_loader']['y_column']]
                    df_tw_curt.drop(labels=[self.config['data_loader']['y_column']], axis=1, inplace=True)
                    self.X_test = df_tw_curt.copy()
                    
                    print(self.X_train)
                    print(self.y_train)
                    print(self.X_test)
                    print(self.y_test)
                    #print(len(df_tw_curt))
                    
                    print(prev_index, curt_index)
                    ##yield (self.preprocessing(df_tw_prev), self.preprocessing(df_tw_curt))
                    prev_index = curt_index
                    curt_index += 1
    
    
        elif self.config['data_loader']['name'] == 'SingleFile':
            print('SingleFile')
            file_name = self.config['data_loader']['file_name']
            all_data = self.load_from_file(path+file_name)
            target = all_data[self.config['data_loader']['y_column']]
            print(target)
            # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state) 
            self.X_train,self.X_test, self.y_train, self.y_test = train_test_split(all_data,target,test_size = 0.2,random_state = 0) 
            #train_y= train_y['label']
            print(len(self.X_train))
            print(len(self.y_train))
            print(len(self.X_test))
            print(len(self.y_test))

            validation_df = all_data.sample(frac = 0.2,replace = False)
            self.y_validation = validation_df[self.config['data_loader']['y_column']]
            validation_df.drop(labels=[self.config['data_loader']['y_column']], axis=1, inplace=True)
            self.X_validation = validation_df.copy()


    @classmethod
    def files_path_dir(self, path):
        data_file_name = os.listdir(path)
        data_file_path = []
        for file in data_file_name:
            data_file_path.append(os.path.join(path, file))
        data_file_path.sort()
        return data_file_path

    def load_from_file(self, path):
        data_chunks = []
        chunk_count = 0
        print('load_from_file.path',path)
        ####usecols=list(feature_names.keys()),
        for chunk in pd.read_csv(path, header=0, sep=',', encoding='UTF-8', engine='python', chunksize=100000):
            chunk_count += 1            
            if(chunk_count%2==0):
                break
            #if(chunk_count==2):
            #   break   
            print('Loading Chunk ' + str(chunk_count))
            data_chunks.append(chunk)
       
        print(len(data_chunks))
        data = pd.concat(data_chunks, axis=0)
        print(len(data))
        del data_chunks
        #df = pd.read_csv(path)
        return data

    def get_test_data(self):
        return self.X_test, self.y_test
    def get_validation_data(self):
        return self.X_validation, self.y_validation       
    def get_train_data(self):
        return self.X_train, self.y_train
    

if __name__ == "__main__":
    print("begin")
    config = {'data_loader': {'name':'TimeSeries', 'data_path':'../dataset'}}
    tet = ModelDataLoader(config=config)
    t= tet.files_path_dir(config=config)
    t.print_t()
    print(t)
    print("Done")
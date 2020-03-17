from base.base_data_loader import BaseDataLoader
import os
from keras.datasets import mnist
from keras.utils import to_categorical
import pandas as pd


class ModelDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ModelDataLoader, self).__init__(config)
        self.config = config
        self.file_list = []

    def load_data(self):
        if self.config['data_loader']['name'] == 'TimeSeries':
            self.files_path_dir(self.config)
            while True:
                try:
                    prev_df = self.load_from_file(next(self.file_list))
                    curr_df = self.load_from_file(next(self.file_list))
                except StopIteration:
                    break
                yield prev_df, curr_df

        elif self.config['data_loader']['name'] == 'SingleFile':
            pass

        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)

    @classmethod
    def files_path_dir(cls, config=None):
        obj = cls(config=config)
        datasets_path = obj.config['data_loader']['data_path']
        assert datasets_path != '', 'Notiece : Data Path is NULL !!!'
        for file in os.listdir(path=datasets_path):
            tmp_path = os.path.join(datasets_path, file)
            if os.path.isdir(tmp_path):
                obj.files_path_dir(tmp_path)
            else:
                obj.file_list.append(file)
                obj.file_list.sort()

    def load_from_file(self, path):
        df = pd.read_csv(path)
        return df

    def get_test_data(self):
        return self.X_test, self.y_test

    def print_t(self):
        print("test")

if __name__ == "__main__":
    config = {'data_loader': {'name':'TimeSeries', 'data_path':'../dataset'}}
    tet = ModelDataLoader(config=config)
    t= tet.files_path_dir(config=config)
    t.print_t()
    print(t)
    print("Done")
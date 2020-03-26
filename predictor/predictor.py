#from base.predictor import Predictor
from keras.models import load_model
import numpy as np
import os

class ModelPredict(object):
    def __init__(self, model, data, config):
        self.config = config
        self.model = model
        self.data = data
        self.predictor()


   
    def predictor(self):
        #get the h5 file 
        reload_model = load_model(self.config['model']['save_model_name'])
        #self.model.summary()
        print("测试")
        print(self.data[0])
        print("测试")
        print(self.data[1])
        predict_test = reload_model.predict(self.data[0])
        predict = np.argmax(predict_test,axis=1)  #axis = 1是取行的最大值的索引，0是列的最大值的索引
        #inverted = np.encoder.inverse_transform([predict])
        print(predict_test[0:3])
        print(np.argmax(predict_test,axis=1))


from base.base_model import BaseModel
from keras.models import Sequential,load_model
import os
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Reshape, Embedding


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.build_model()

    def build_model(self):
        #self.input_data = Input(name='the_input', shape=(28, 28), dtype='float32')
        self.model = Sequential()
        self.model.add(Reshape((16, 16, 1), input_shape=(16, 16)))
        #self.model.add(pad_sequence())
        ##self.model.add(Embedding(input_dim=16*16+1, output_dim=128,input_shape=(16, 16)))
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())#降维
        self.model.add(Dense(81, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3))#分类的类数
        self.model.add(Activation('softmax'))
        self.model.summary()
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=self.config['model']['optimizer'],
            metrics=['acc'],
        )

    def save_model(self):
        if not os.path.isdir(self.config['model']['save_model_path']):
            os.makedirs(self.config['model']['save_model_path'])
        #self.model.save("model.h5") 
        #self.model.save_weights("model_weights.h5")       
        
        self.model.save(self.config['model']['save_model_name'])
        self.model.save_weights(self.config['model']['save_weights_name'])
    
    def load_models(self):
        #models = load_model(self.config['model']['save_model_name'])
        self.model.load_model(self.config['model']['save_model_name'])
        self.model.summary()
    def load_model_weights(self):
        #models = load_model(self.config['model']['save_model_name'])
        self.model.load_weights(self.config['model']['save_weights_name'])
        self.model.summary()
    
    
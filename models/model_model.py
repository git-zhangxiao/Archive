from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Reshape, Embedding


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.build_model()

    def build_model(self):
        #self.input_data = Input(name='the_input', shape=(28, 28), dtype='float32')
        self.model = Sequential()
        #self.model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
        self.model.add(Embedding(input_dim=28*28, output_dim=128, input_shape=(28, 28)))
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(81, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        self.model.summary()
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=self.config['model']['optimizer'],
            metrics=['acc'],
        )


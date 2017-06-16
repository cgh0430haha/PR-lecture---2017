# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:07:43 2017

@author: geunho
"""

import keras
from keras import regularizers
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import plot_model
import h5py

tf.logging.set_verbosity(tf.logging.INFO)


TRAINING = "train_1_100.csv"
TEST = "test_1_100.csv"


batch_size = 100
num_classes =2
epochs = 50 #50
img_rows, img_cols = 6,6

dataframe=pd.read_csv(TRAINING,engine='python')
dataset=dataframe.values
dataset=dataset.astype('float32')

x,y = dataset[:,:-2], dataset[:,2253:]


# Load datasets.
xy = np.loadtxt(TRAINING,delimiter=',',dtype=np.float32)
x = xy[:, 0:-2]
y = xy[:, -2: ]
xy = np.loadtxt(TEST,delimiter=',',dtype=np.float32)
eval_data = xy[:, 0:-2]
eval_labels = xy[:, -2: ]


#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train,y_train=x,y
x_test,y_test = eval_data,eval_labels  

print (x_train.shape[0])
x_train = np.reshape(x_train, (x_train.shape[0],3,751,1))
x_test = np.reshape(x_test, (x_test.shape[0],3,751,1))


# Make sure the shape and data are OK
print(x.shape, x, len(x))
print(y.shape, y, len(y))
#print(y_data.shape, y_data)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),padding='same',input_shape=(3,751,1),
                 kernel_regularizer=regularizers.l2(0.0001)) )                
model.add(MaxPooling2D(pool_size=(1,2) ) ) # (vertical, horizontal). (2, 2) 
                                            # AveragePooling2D
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))

model.add(Conv2D(64, kernel_size=(3,3), padding='same',kernel_regularizer=regularizers.l2(0.0001)) )
model.add(MaxPooling2D(pool_size=(1,2) ) ) # (vertical, horizontal). (2, 2) 
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Conv2D(128, kernel_size=(3,3), padding='same',kernel_regularizer=regularizers.l2(0.0001)) )
model.add(MaxPooling2D(pool_size=(1,2) ) ) # (vertical, horizontal). (2, 2) 
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Conv2D(128, kernel_size=(3,3), padding='same',kernel_regularizer=regularizers.l2(0.0001)) )
model.add(MaxPooling2D(pool_size=(1,2) ) ) # (vertical, horizontal). (2, 2) 
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Conv2D(128, kernel_size=(3,3), padding='same',kernel_regularizer=regularizers.l2(0.0001)) )
model.add(MaxPooling2D(pool_size=(1,2) ) ) # (vertical, horizontal). (2, 2) 
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Conv2D(128, kernel_size=(3,3), padding='same',kernel_regularizer=regularizers.l2(0.0001)) )
model.add(MaxPooling2D(pool_size=(1,2) ) ) # (vertical, horizontal). (2, 2) 
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Conv2D(128, kernel_size=(3,3), padding='same',kernel_regularizer=regularizers.l2(0.0001)) )
model.add(MaxPooling2D(pool_size=(1,2) ) ) # (vertical, horizontal). (2, 2) 
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
#model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same') )
#model.add(MaxPooling2D(pool_size=(2,2) ) )
#model.add(Dropout(0.25) )
model.add(Flatten() )
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.0001)) )
#model.add(Dropout(0.5) )
model.add(Dense(num_classes, activation='softmax', name='predictions') )

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD() #lr=0.01, momentum=0.9
              ,metrics=['accuracy'])

plot_model(model, to_file='model.png', show_shapes=True)
#optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=False),
#keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #0.7
#keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)         
#keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0) #0.6861
#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#SGD() #0.705


print(model.layers)
print(model.inputs)
print(model.outputs)
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output_shape)
tfBoard=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
              write_graph=True, write_images=True, embeddings_freq=0, 
              embeddings_layer_names=None, embeddings_metadata=None)
csv_logger = keras.callbacks.CSVLogger('training.log', separator=',', append=True)


model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1,validation_data=(x_test, y_test),callbacks=[tfBoard, csv_logger])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss', score[0])
print('Test accuracy', score[1])

for i in range(0,9):
    index = 40*i
    score_person = model.evaluate(x_test[index:index+40], y_test[index:index+40], verbose=0)
    print('Person_%d Test loss : %f' % (i, score_person[0]))
    print('Person_%d Test accuracy : %f' % (i, score_person[1]))





'''
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
'''

















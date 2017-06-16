# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:04:56 2017

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
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import plot_model
from vis.visualization import visualize_saliency
from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from vis.utils import utils
from pylab import plot, show
import cv2
from attentionMap import visualize_saliency, get_num_filters, visualize_cam

tf.logging.set_verbosity(tf.logging.INFO)


TRAINING = "train_8_30.csv"
TEST = "test_8_30.csv"
batch_size = 100
num_classes =2
epochs =5 #50
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


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()

json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")


# evaluate loaded model on test data
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print(model.predict(x_test[1:5]))


for idx, layer in enumerate(model.layers) :
    print(idx)
    print(layer.name)
    print('~~~~')


layer_name = 'conv2d_4'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer_idx]))
filters = list(filters)
# Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
vis_images = []

'''
for idx in range(0,2):#filters:
    img = visualize_activation(model, 30, filter_indices=idx) #30 == predictions 
    #img = utils.draw_text(img, str(idx))
    # Remove the first axis.
    out = np.squeeze(img)
    # We want to move axis=2 (64) to 0 so that we can treat `out` as a slices of images with (h, w) values.
    out = np.moveaxis(out, 0, 1)
    vis_images.append(out)
    print(vis_images)


#time = range(0, 751)
#plot(output[0,:], marker='o')
#show()

plt.figure()    
output=np.squeeze(img)
plt.imshow(output,cmap='gray')# cmap='gray'
plt.colorbar()
plt.title(layer_name)
'''



#plt.show()
'''
input("Press Enter to continue...")

# Generate stitched image palette with 8 cols.
vis_images=np.squeeze(vis_images)
stitched = utils.stitch_images(vis_images, cols=8)    
plt.axis('off')
plt.plot(stitched)
#plt.imshow(stitched)
plt.title(layer_name)
plt.show()

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

heatmaps = []
opposite_heatmaps = []
opposite_pred_class=0;
person = 1112 #1112 134 120+29
for i in range(person, person+2): 
    #x1 = np.expand_dims(x2, axis=0)
    #x = np.reshape(x2, (-1,3,751,1))
    print(x_train[person:person+2,:,:,:].shape)
    a=x_train[i,:,:,:]
    #print(x2.shape, x2, len(x2))
    #layerInput = np.squeeze(a, axis=(0,))
    
    layerInput = a    
    #layerInput = np.expand_dims(a, axis=0)
    print(layerInput.shape)

    #print(a.shape)
    #seed_img = utils.load_img(i, target_size=(224, 224))
    #x = np.expand_dims(img_to_array(seed_img), axis=0)
    #x = preprocess_input(x)
    AAA = model.predict_classes(np.expand_dims(x_train[i,:,:,:], axis=0))
    print(AAA)
    print(y_train[i,])
    #pred_class = np.argmax( model.predict(x_test[0,:,:,:] ) )
    pred_class = np.argmax( AAA )
    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    if pred_class==1 :
        opposite_pred_class=0;
    else :
        opposite_pred_class=1;

    #visualize_saliency     visualize_cam
    heatmap = visualize_cam(model, layer_idx, [pred_class], layerInput)        
    heatmaps.append(heatmap)
    opposite_heatmap = visualize_cam(model, layer_idx, [opposite_pred_class], layerInput)
    opposite_heatmaps.append(opposite_heatmap) # *5  --> saliency



trial1=heatmaps[0]; trial2=heatmaps[1]
trial1_opposite=opposite_heatmaps[0]; trial2_opposite=opposite_heatmaps[1]
# expand 2d -> 3d for visualize_cam
trial1 = np.expand_dims(trial1, axis=0)
trial2 = np.expand_dims(trial2, axis=0)
trial1_opposite = np.expand_dims(trial1_opposite, axis=0)
trial2_opposite = np.expand_dims(trial2_opposite, axis=0)


plt.figure()
idx2=0
plt.plot(x_train[person,0,:,:], label="C3")
plt.plot(x_train[person,1,:,:], label="Cz")
plt.plot(x_train[person,2,:,:], label="C4")
plt.legend(loc=2);plt.title('raw input (person[1],trial[6])');
plt.show()

plt.figure()
plt.plot(trial1[0,0,:], label="C3")
plt.plot(trial1[0,1,:], label="Cz")
plt.plot(trial1[0,2,:], label="C4")
plt.legend(loc=2);plt.title('C3,Cz,C4 heatmap (person[1],trial[6])');
plt.show()

plt.figure()
plt.plot(x_train[person,0,:,:], label="raw input")
plt.plot(trial1[0,0,:], label="C3_saliency")
plt.plot(trial1_opposite[0,0,:], '--', label="C3_oppositePred")
plt.legend(loc=2);plt.title('C3 heatmap + raw input (person[1],trial[6])');
plt.show()

plt.figure()
plt.plot(x_train[person,1,:,:], label="raw input")
plt.plot(trial1[0,1,:], label="Cz_saliency")
plt.plot(trial1_opposite[0,0,:], '--', label="Cz_oppositePred")
plt.legend(loc=2);plt.title('Cz heatmap + raw input (person[1],trial[6])');
plt.show()

plt.figure()
plt.plot(x_train[person,2,:,:], label="raw input")
plt.plot(trial1[0,2,:], label="C4_saliency")
plt.plot(trial1_opposite[0,0,:], '--', label="C4_oppositePred")
plt.legend(loc=2);plt.title('C4 heatmap + raw input (person[1],trial[6])');
plt.show()



'''
plt.figure()
idx2=0
plt.plot(right[:,0], label="0")
plt.plot(right[:,1], label="1")
plt.plot(right[:,2], label="2")
plt.legend(loc=2);plt.title('right');
plt.show()


plt.figure()
idx2=0
plt.plot(left)
plt.plot(right)
plt.legend(loc=2)
plt.show()
'''

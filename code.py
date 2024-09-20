import pandas as pd
import PIL
from PIL import Image
from skimage import io
import tensorflow as tf
import pathlib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
train=ImageDataGenerator(rescale=1/255)
valid=ImageDataGenerator(rescale=1/255)
train_ds=train.flow_from_directory('C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\train',
                                  target_size=(256,256), batch_size=64,
                                  class_mode='binary')
valid_ds=train.flow_from_directory('C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\validate',
                                  target_size=(256,256),
                                   batch_size=64,
                                  class_mode='binary')
test_ds=train.flow_from_directory('C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\test',
                                  target_size=(256,256),
                                   batch_size=64,
                                  class_mode='binary')
train_ds[0]
img=image.load_img('C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\train\\cyclone\\0.jpg')
plt.imshow(img)
train_ds[0][0][0][0]
plt.imshow(train_ds[0][0][0])
train_ds.class_indices
base1 = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base1.trainable = False
model1 = tf.keras.Sequential([
    base1,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.Accuracy(),
                                                                                           tf.keras.metrics.Precision(),tf.keras.metrics.MeanAbsoluteError()])
history1=model1.fit(train_ds,
                 epochs=30)
model1.predict(test_ds)
import os
path='C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\test\\cyclone'
for i in os.listdir(path):
    img= image.load_img(path+'\\'+i, target_size=(256,256))
    plt.imshow(img)
    plt.show()
    
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val= model1.predict(images)
    #binary_predictions = (val >= threshold).astype(int)
    print(val)
    if val==0:
        print('cyclone')
    else:
        print('non cyclone')
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load pre-trained Xception model without fully connected layers
base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False

# Choose the layer from which you want to visualize the feature maps
layer_name = 'conv1'  # Example: layer name from Xception

# Extract the desired layer's output
selected_layer_output = base_model.get_layer(layer_name).output

# Create a model that outputs the feature maps of the selected layer
feature_map_model = Model(inputs=base_model.input, outputs=selected_layer_output)

# Load and preprocess the image
img_path = 'C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\train\\cyclone\\0.jpg'
img = image.load_img(img_path, target_size=(256, 256))  # Adjust target size to match your input shape
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

# Get the feature maps for the input image
feature_maps = feature_map_model.predict(img_array)

# Visualize the feature maps
plt.figure(figsize=(16, 8))
for i in range(feature_maps.shape[-1]):
    plt.subplot(4, 8,i+1)  # Adjust the subplot grid as needed
    plt.imshow(feature_maps[0, :, :, i], cmap='cividis')  # Choose a colormap as needed
    plt.axis('off')
plt.show()
base2=tf.keras.applications.nasnet.NASNetLarge(include_top=False,weights='imagenet',input_tensor=Input(shape=(256, 256, 3)))
base2.trainable = False
model2 = tf.keras.Sequential([
    base2,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.Accuracy(),
                                                                                           tf.keras.metrics.Precision(),tf.keras.metrics.MeanAbsoluteError()])
history2=model2.fit(train_ds,
                  epochs=30)
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load pre-trained Xception model without fully connected layers
base_model = tf.keras.applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False

# Choose the layer from which you want to visualize the feature maps
layer_name = 'stem_conv1'  # Example: layer name from Xception

# Extract the desired layer's output
selected_layer_output = base_model.get_layer(layer_name).output

# Create a model that outputs the feature maps of the selected layer
feature_map_model = Model(inputs=base_model.input, outputs=selected_layer_output)

# Load and preprocess the image
img_path = 'C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\train\\cyclone\\0.jpg'
img = image.load_img(img_path, target_size=(256, 256))  # Adjust target size to match your input shape
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.nasnet.preprocess_input(img_array)

# Get the feature maps for the input image
feature_maps = feature_map_model.predict(img_array)

# Visualize the feature maps
plt.figure(figsize=(16, 8))
for i in range(feature_maps.shape[-1]):
    plt.subplot(4, 8,i+1)  # Adjust the subplot grid as needed
    plt.imshow(feature_maps[0, :, :, i], cmap='magma')  # Choose a colormap as needed
    plt.axis('off')
plt.show()
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,Average
m1=load_model('C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\mobile.h5')
m2=load_model('C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\nasnet1.h5')

m1=Model(inputs=m1.inputs,
        outputs=m1.outputs,
        name='dense')
m2=Model(inputs=m2.inputs,
        outputs=m2.outputs,
        name='nasnet')

models=[m1,m2]
model_input=Input(shape=(256,256,3))
model_output=[model(model_input) for model in models]
e_out=Average()(model_output)
ensemble=Model(inputs=model_input, outputs=e_out, name='ensemble')
ensemble.summary()
ensemble.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.Accuracy(),
                                                                                           tf.keras.metrics.Precision(),tf.keras.metrics.MeanAbsoluteError()])
e_history=ensemble.fit(train_ds,
                      validation_data=valid_ds,
                      epochs=25)
evals=ensemble.evaluate(test_ds)
act=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]
pred=ensemble.predict(test_ds)
c=0
nc=0
for i in pred:
    if i==0:
        c+=1
    else:
        nc+=1
print('Total Cyclone images:',c)
print('Total Non Cyclone images:',nc)

predi=pred.flatten()
l=[]
for i in predi:
    if i==0:
        l.append(0)
    else:
        l.append(1)
data = {'Actual':    act,
        'Predicted': l
        }
l
import os
threshold = 0.5
path='C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\validate\\cyclone'
for i in os.listdir(path):
    img= image.load_img(path+'\\'+i, target_size=(256,256))
    plt.imshow(img)
    plt.show()
    
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val= ensemble.predict(images)
    #binary_predictions = (val >= threshold).astype(int)
    #print(val)
    if val==0:
        print('cyclone')
    else:
        print('non cyclone')
import os
threshold = 0.5
path='C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\validate\\non cyclone'
for i in os.listdir(path):
    img= image.load_img(path+'\\'+i, target_size=(256,256))
    plt.imshow(img)
    plt.show()
    
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val= ensemble.predict(images)
    #binary_predictions = (val >= threshold).astype(int)
    #print(val)
    if val==0:
        print('cyclone')
    else:
        print('non cyclone')
import os
threshold = 0.5
path='C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\test\\cyclone'
for i in os.listdir(path):
    img= image.load_img(path+'\\'+i, target_size=(256,256))
    plt.imshow(img)
    plt.show()
    
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val= ensemble.predict(images)
    #binary_predictions = (val >= threshold).astype(int)
    #print(val)
    if val==0:
        print('cyclone')
    else:
        print('non cyclone')
import os
threshold = 0.5
path='C:\\Users\\SYSTEMS\\OneDrive\\Desktop\\data\\test\\non cyclone'
for i in os.listdir(path):
    img= image.load_img(path+'\\'+i, target_size=(256,256))
    plt.imshow(img)
    plt.show()
    
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val= ensemble.predict(images)
    #binary_predictions = (val >= threshold).astype(int)
    #print(val)
    if val==0:
        print('cyclone')
    else:
        print('non cyclone')

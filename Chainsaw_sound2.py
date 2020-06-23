import keras
from keras.layers import Activation,Dense,Dropout,Conv2D,Flatten,MaxPooling2D,Convolution2D,GlobalAveragePooling2D
import schedule
from keras.callbacks import LearningRateScheduler
from keras.models import  Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import  matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import glob
from keras.layers.normalization import BatchNormalization
from scipy.io.wavfile import read
import wave, os, glob
from keras.layers import Dense
from keras import backend as K
import keras

label=[]
for k in range(1,2):
    path = (r'C:\Users\Sargis\PycharmProjects\ Chainsaw_Sound_Recognition\venv\Include\wavSound\fold' + str(k) )
    for filename in glob.glob(os.path.join(path, '*.wav')):
        x=filename.split('.')[1]
        y=x.split('\\')[-1]
        label.append(y)




D=[]
p=0
for k in range(1,2):
    path = (r'C:\Users\Sargis\PycharmProjects\ Chainsaw_Sound_Recognition\venv\Include\wavSound\fold' + str(k) )
    for filename in glob.glob(os.path.join(path, '*.wav')):
        y, sr = librosa.load(filename)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        if ps.shape != (128,22):continue
        D.append((ps,label[p]))
        p=p+1
dataset=D


print(len(dataset))
random.shuffle(dataset)
train=dataset[:850]
test=dataset[850:]


x_train,y_train=zip(*train)
x_test,y_test=zip(*test)
x_train=np.array(x_train)
nsamples, nx, ny = x_train.shape
x_train = x_train.reshape((nsamples,nx*ny))
from  sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_train=scaler.fit_transform(x_train)

scaler2= StandardScaler()
x_test=np.array(x_test)
nsamples, nx, ny = x_test.shape
x_test = x_test.reshape((nsamples,nx*ny))
x_test=scaler2.fit_transform(x_test)
y_train=np.array(y_train,dtype=int)
y_train[y_train==0]=0
y_train[y_train==1]=1

y_test=np.array(y_test,dtype=int)
y_test[y_test==0]=0
y_test[y_test==1]=1

y_train=np_utils.to_categorical(y_train,2)
y_test=np_utils.to_categorical(y_test,2)



x_train=np.array([x.reshape((128,22,1)) for x in x_train])
x_test=np.array([x.reshape((128,22,1)) for x in x_test])
print()

model=Sequential()
input_shape=(128,22,1)


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=2, input_shape=input_shape))
model.add(MaxPooling2D(pool_size=2))
model.add(Activation('relu'))

# model.add(Conv2D(filters=32, kernel_size=2))
# model.add(Activation('relu'))


model.add(Conv2D(filters=32, kernel_size=2))
model.add(MaxPooling2D(pool_size=2))
model.add(Activation('relu'))


# model.add(Conv2D(filters=128, kernel_size=2))
# model.add(Activation('relu'))
# model.add(BatchNormalization( ))



model.add(Conv2D(filters=64, kernel_size=2))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Activation('relu'))


model.add(GlobalAveragePooling2D())
model.add(Dense(2, activation='softmax'))

def scheduler(epochs,lr):

    if epochs % 10==0 and epochs:
        lr=lr*0.1
        return lr
    return lr
  # elif epochs ==10:
  #   return 0.00023
  # elif epochs ==64:
  #   return 0.00023
  #   else:
  #       return 0.0001
optimizer = keras.optimizers.Adam()
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=[tf.keras.metrics.Precision()])
history=model.fit(x=x_train,y=y_train, epochs=30 ,batch_size=64,validation_data=(x_test,y_test),callbacks=[callback])
print(history.history.keys())
loss_train=np.array(history.history['loss'])
loss_test=np.array(history.history['val_loss'])
plt.plot(loss_train)
plt.plot(loss_test)
plt.show()
print(history.history['lr'])
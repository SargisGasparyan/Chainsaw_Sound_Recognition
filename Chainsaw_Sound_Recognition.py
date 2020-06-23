import keras
from keras.layers import Activation,Dense,Dropout,Conv2D,Flatten,MaxPooling2D,Convolution2D,GlobalAveragePooling2D
from keras.models import  Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

import  matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import glob
from scipy.io.wavfile import read
import wave, os, glob



label=[]
for k in range(1,11):
    path = (r'C:\Users\Sargis\PycharmProjects\ Chainsaw_Sound_Recognition\venv\Include\audio\fold' + str(k) )
    for filename in glob.glob(os.path.join(path, '*.wav')):
        x=filename.split(' ')[1]
        y=x.split('\\')[-1]
        label.append(y)



D=[]
p=0
for k in range(1,11):
    path = (r'C:\Users\Sargis\PycharmProjects\ Chainsaw_Sound_Recognition\venv\Include\audio\fold' + str(k) )
    for filename in glob.glob(os.path.join(path, '*.wav')):
        y, sr = librosa.load(filename)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        if ps.shape != (128,22):continue
        D.append((ps,label[p]))
        p=p+1
# filename=r'C:\Users\Sargis\PycharmProjects\ Chainsaw_Sound_Recognition\venv\Include\audio\fold1\2 (1) 0042.wav'
# # y, sr = librosa.load(filename)
# # print(len(y))
# # print(sr)
# y=11025
# sr=22050
# ps=librosa.feature.melspectrogram(y=y,sr=sr)
# print(plt.show())
#

dataset=D
print(len(dataset))
random.shuffle(dataset)
train=dataset[:4200]
test=dataset[4200:]

x_train,y_train=zip(*train)
x_test,y_test=zip(*test)
y_train=np.array(y_train,dtype=int)
y_train[y_train==1]=0
y_train[y_train==2]=1

y_test=np.array(y_test,dtype=int)
y_test[y_test==1]=0
y_test[y_test==2]=1
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train,2)
y_test=np_utils.to_categorical(y_test,2)



x_train=np.array([x.reshape((128,22,1)) for x in x_train])
x_test=np.array([x.reshape((128,22,1)) for x in x_test])
print()

model=Sequential()
input_shape=(128,22,1)


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))




model.add(GlobalAveragePooling2D())
model.add(Dense(2, activation='softmax'))


# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(x=x_train,y=y_train,epochs=10,batch_size=64,validation_data=(x_test,y_test))
score=model.evaluate(x=x_test,y=y_test)
score2=model.evaluate(x=x_train,y=y_train)



print(score[0])
print(score[1])
print(score2[0])
print(score2[1])

path2 = (r'C:\Users\Sargis\PycharmProjects\ Chainsaw_Sound_Recognition\venv\Include\1wr (628).wav')
y, sr = librosa.load(path2)
ps2 = librosa.feature.melspectrogram(y=y, sr=sr)
tt=ps2.reshape((1,128,22,1))
pred = model.predict_classes(tt)
print(pred)

import os
import cv2
import numpy as np
from keras.models import load_model

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten,BatchNormalization
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


path = os.path.join(os.getcwd(),"FER\Data1")

dirs = os.listdir(path)

images = []
labels = []

for i in dirs:
    new_p = os.path.join(path,i)
    for j in os.listdir(new_p):
        image = cv2.imread(os.path.join(new_p,j),cv2.IMREAD_ANYCOLOR)
        image = cv2.resize(image,(48,48))
        images.append(image)
        labels.append(dirs.index(i))

img = np.array(images)
labls = np.array(labels)

img = img/255
img = np.expand_dims(img,axis=3)


labels_encoded=to_categorical(labls,num_classes=7)


labls = np.expand_dims(labels, axis=1)

x_train,x_test,y_train,y_test = train_test_split(img,labels_encoded,test_size=0.5)

model = Sequential()

model.add(Conv2D(32,(3,3),strides=(1,1),padding="same",input_shape=(48,48,1),activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64,(3,3),strides=(1,1),padding="same",activation="relu"))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),strides=(1,1),padding="same",activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Conv2D(256,(3,3),strides=(1,1),padding="same",activation="relu"))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(7,activation="softmax"))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

history=model.fit(x_train,y_train,batch_size=32,validation_data=(x_test,y_test),epochs=100)

model.save("model.h5")

y_p = model.predict(x_test)

print(confusion_matrix(np.argmax(y_test,axis=1),np.argmax(y_p,axis=1)))





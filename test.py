import numpy as np
import cv2
from keras.models import load_model


codes = {0: "Angry", 1: "Contempt", 2: "Disgusted", 3: "Fearful", 4: "Happy", 5: "Sad", 6: "Surprised"}

model = load_model("model.h5")
face_cascade = cv2.CascadeClassifier('FER/haarcascade_frontalface_default.xml')
image = cv2.imread('image.png')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

pred = []
for (x, y, w, h) in faces:
    roi = gray[x:x+w,y:y+h]
    img = cv2.resize(roi,(48,48), interpolation=cv2.INTER_LINEAR)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img,axis=3)
    pred.append(np.argmax(model.predict(img)))

if len(faces) > 1:
    i = 0
    for f in faces:
        face = faces[i]
        (x, y, w, h) = face

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, codes[pred[i]], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 255), 1)
        i = i + 1
else:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, codes[pred[0]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 255), 1)

cv2.imshow('image',image)
cv2.waitKey()
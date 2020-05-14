import os
import sys
from cv2 import cv2
import json
import numpy as np
import tensorflow as tf

MODEL_PATH =os.path.join(os.getcwd(),'mask_classifier.h5')
MODEL_PATH2 =os.path.join(os.getcwd(),'mask.json')

json_file=open(MODEL_PATH2,'r')
loaded_model_json=json_file.read()
json_file.close()

model=tf.keras.models.model_from_json(loaded_model_json)
model.load_weights(MODEL_PATH)

cascPath="haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascPath)
video_capture=cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)


while True:
  ret,frame=video_capture.read()
  if frame is not None:
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    image=cv2.resize(frame,(64,64))
    image=np.array(image,dtype='float32')/255.0
    image=np.expand_dims(image,axis=0)
    pred=model.predict(image)
    #pred=np.amax(pred,axis=1)
    print(pred)
    if np.argmax(pred)==1:
        text='no mask'
    else:
        text='mask'
    
    faces=faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        #flags=cv2.CV_HAAR_SCALE_IMAGE  
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.putText(frame , text , (x , y) , cv2.FONT_HERSHEY_SIMPLEX , 0.45 , (255,255,255), 2)
    cv2.imshow('Video', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows() 



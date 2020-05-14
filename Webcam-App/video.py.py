from cv2 import cv2
import os
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

video = cv2.VideoCapture(0)
path_model=os.path.join(os.getcwd(),'res10_300x300_ssd_iter_140000.caffemodel')
path_protxt=os.path.join(os.getcwd(),'deploy.prototxt.txt')
video.set(cv2.CAP_PROP_FRAME_WIDTH,640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
net = cv2.dnn.readNetFromCaffe(path_protxt ,path_model)

MODEL_PATH =os.path.join(os.getcwd(),'model3.h5')
MODEL_PATH2 =os.path.join(os.getcwd(),'model3.json')

json_file=open(MODEL_PATH2,'r')
loaded_model_json=json_file.read()
json_file.close()

model=tf.keras.models.model_from_json(loaded_model_json)
model.load_weights(MODEL_PATH)

conf = 0.70
no_conf = 1.99

while True:
	check, frame = video.read()
	blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0 ,(300,300) ,(104.0, 177.0, 123.0))
	(h,w) = frame.shape[:2]
	counter=0
	net.setInput(blob)
	detections = net.forward()
	
	for i2 in range(0 , detections.shape[2]):
		confidence = detections[0,0,i2,2]

		if (confidence > conf) and (confidence < no_conf) :
			box = detections[0,0,i2,3:7]*np.array([w,h,w,h])
			(startX , startY , endX , endY) = box.astype("int")
			(startX, startY) = (max(0, startX),max(0, startY))
			(endX, endY) = min(w - 1, endX), min(h - 1, endY)

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			(mask, withoutMask) = model.predict(face)[0]
			print(mask,withoutMask)
			if mask < withoutMask:
				label="Mask" 
				color=(0,255,0)  
			else :
				label="No Mask"
				color=(0,0,255)


			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			cv2.rectangle(frame , (startX , startY) , (endX,endY) , color , 2)
			cv2.putText(frame , label , (startX , startY) , cv2.FONT_HERSHEY_SIMPLEX , 0.45 , (255,255,255), 2)
	
	cv2.imshow('Video window', frame)

	# press 'q' on keyboard to exit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video.release()
cv2.destroyAllWindows()
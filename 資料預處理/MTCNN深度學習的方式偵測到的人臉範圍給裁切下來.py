import numpy as np
import pandas as pd
import keras
import tensorflow
from tensorflow.keras.models import load_model
import gc
import matplotlib.pyplot as plt
import cv2
import os
import dlib
from IPython.display import clear_output
from mtcnn import MTCNN
import cv2
 
detector = MTCNN()
image = cv2.imread('C:\\Users\\admin\\Desktop\\2000\\original0.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
detections = detector.detect_faces(image)
x, y, width, height = detections[0]['box']
x1,y1,x2,y2 = x-10,y+10,x-10 +width + 20,y+10+height
face = image[y1:y2, x1:x2]
#face = cv2.resize(face, (170, 170), interpolation=cv2.INTER_AREA) #if shape is > 120x120
face = cv2.resize(face, (120, 120), interpolation=cv2.INTER_LINEAR)
plt.imshow(face)
plt.show()

def extract_faces(source,destination,detector):
    counter = 0
    for dirname, _, filenames in os.walk(source):
        for filename in filenames:
            try:
                image = cv2.imread(os.path.join(dirname, filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(image)
                x, y, width, height = detections[0]['box']
                x1,y1,x2,y2 = x-10,y+10,x-10 +width + 20,y+10+height
                face = image[y1:y2, x1:x2]
                face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_LINEAR)
                plt.imsave(os.path.join(destination,filename),face)
                clear_output(wait=True)
                print("Extraction progress: "+str(counter)+"\\"+str(len(filenames)-1))
            except:
                pass
            counter += 1
            
detector = MTCNN()
extract_faces('C:\\Users\\admin\\Desktop\\2000\\', 'C:\\Users\\admin\\Desktop\\2001\\',detector)
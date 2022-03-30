#!/usr/bin/env python
# coding: utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf


def create_dataset(path):
    images = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            image = cv2.imread(os.path.join(dirname, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype('float32')
            image /= 255.0
            images.append(image)
    images = np.array(images)
    return images




faces_1 = create_dataset('/content/drive/MyDrive/120face')
faces_2 = create_dataset('/content/drive/MyDrive/120franky')


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.imshow(faces_1[1])
plt.show()



print("Total President Trump face's samples: ",len(faces_1))
print("Total President Biden face's samples: ",len(faces_2))




X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(faces_1, faces_1, test_size=0.20, random_state=0)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(faces_2, faces_2, test_size=0.15, random_state=0)



X_train_a[0].shape





#Making encoder:

input_img = layers.Input(shape=(120, 120, 3))
x = layers.Conv2D(256,kernel_size=5, strides=2, padding='same',activation='relu')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(512,kernel_size=5, strides=2, padding='same',activation='relu')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(1024,kernel_size=5, strides=2, padding='same',activation='relu')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(9216)(x)
encoded = layers.Reshape((3,3,1024))(x)

encoder = keras.Model(input_img, encoded,name="encoder")
encoder.summary()
#encoder.load_weights('/content/drive/MyDrive/models/encoder.h5')
encoder.save_weights("/content/drive/MyDrive/models/encoderA.h5")




tf.keras.utils.plot_model(encoder, show_shapes=True, dpi=64)




#Making decoder:
decoder_input= layers.Input(shape=((3,3,1024)))
x = layers.Conv2D(1024,kernel_size=5, strides=2, padding='same',activation='relu')(decoder_input)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(512,kernel_size=5, strides=2, padding='same',activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(256,kernel_size=5, strides=2, padding='same',activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(np.prod((120, 120, 3)))(x)
decoded = layers.Reshape((120, 120, 3))(x)

decoder = keras.Model(decoder_input, decoded,name="decoder")
decoder.summary()
#decoder.load_weights('/content/drive/MyDrive/models/decoder.h5')
decoder.save_weights("/content/drive/MyDrive/models/decoderA.h5")



tf.keras.utils.plot_model(decoder, show_shapes=True, dpi=64)



#Making autoencoder
auto_input = layers.Input(shape=(120,120,3))
encoded = encoder(auto_input)
decoded = decoder(encoded)

autoencoder = keras.Model(auto_input, decoded,name="autoencoder")
#autoencoder.load_weights("/content/drive/MyDrive/models/autoencoder_A.hdf5")
autoencoder.compile(optimizer=keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss='mae')
autoencoder.summary()




tf.keras.utils.plot_model(autoencoder, show_shapes=True, dpi=64)


checkpoint1 = ModelCheckpoint("/content/drive/MyDrive/models/autoencoder_A.hdf5", monitor='val_loss', verbose=1,save_best_only=True, mode='auto', period=1)
history1 = autoencoder.fit(X_train_a, X_train_a,epochs=1500,batch_size=512,shuffle=True,validation_data=(X_test_a, X_test_a),callbacks=[checkpoint1])





plt.plot(history1.history['loss'], label='Training Loss')
plt.plot(history1.history['val_loss'], label='Validation Loss')
plt.legend()





autoencoder_a = load_model("/content/drive/MyDrive/models/autoencoder_A.hdf5")
autoencoder_a.evaluate(X_test_a, X_test_a)



output_image = autoencoder_a.predict(np.array([X_test_a[30]]))



get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.imshow(X_test_a[30])
plt.show()



get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.imshow(output_image[0])
plt.show()




# TO LOAD ONLY THE ENCODER A
encoder_a = keras.Model(autoencoder_a.layers[1].input, autoencoder_a.layers[1].output)
# TO LOAD ONLY THE DECODER A
decoder_a = keras.Model(autoencoder_a.layers[2].input, autoencoder_a.layers[2].output)



input_test = encoder_a.predict(np.array([X_test_a[30]]))
output_test = decoder_a.predict(input_test)
output_test = decoder_a.predict(input_test)



get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.imshow(output_test[0])
plt.show()




import gc
gc.collect()



checkpoint2 = ModelCheckpoint("/content/drive/MyDrive/models/autoencoder_B.hdf5", monitor='val_loss', verbose=1,save_best_only=True, mode='auto', period=1)
history2 = autoencoder.fit(X_train_b, X_train_b,epochs=1500,batch_size=512,shuffle=True,validation_data=(X_test_b, X_test_b),callbacks=[checkpoint2])




plt.plot(history2.history['loss'], label='Training Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.legend()




autoencoder_b = load_model("/content/drive/MyDrive/models/autoencoder_B.hdf5")
autoencoder_b.evaluate(X_test_b, X_test_b)




output_image = autoencoder_b.predict(np.array([X_test_b[0]]))




get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.imshow(X_test_b[0])
plt.show()



get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.imshow(output_image[0])
plt.show()


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
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



get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
image = cv2.imread('/content/drive/MyDrive/frame/original120.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()



get_ipython().system('pip install mtcnn')




from mtcnn import MTCNN
import cv2
 
detector = MTCNN()
image = cv2.imread('/content/drive/MyDrive/myface0606/original0.png')
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
                face = cv2.resize(face, (120, 120), interpolation=cv2.INTER_LINEAR)
                plt.imsave(os.path.join(destination,filename),face)
                clear_output(wait=True)
                print("Extraction progress: "+str(counter)+"/"+str(len(filenames)-1))
            except:
                pass
            counter += 1




detector = MTCNN()
extract_faces('C:\\Users\\admin\\Desktop\\2000\\', 'C:\\Users\\admin\\Desktop\\2001\\',detector)



get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
image = cv2.imread('/content/drive/MyDrive/120face/1001.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()




autoencoder_a = load_model("/content/drive/MyDrive/models/autoencoder_A.hdf5")
autoencoder_b = load_model("/content/drive/MyDrive/models/autoencoder_B.hdf5")




# LOADING THE ENCODER A
encoder_a = keras.Model(autoencoder_a.layers[1].input, autoencoder_a.layers[1].output)
# LOADING THE DECODER B
decoder_b = keras.Model(autoencoder_b.layers[2].input, autoencoder_b.layers[2].output)




def face_transform(source,destination,encoder,decoder):
    counter = 0
    for dirname, _, filenames in os.walk(source):
        for filename in filenames:
            # load the image
            try:
                image = cv2.imread(os.path.join(source, filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype('float32')
                image /= 255.0
                image = encoder.predict(np.array([image]))
                image = decoder.predict(image)
                image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                image = image.astype(np.uint8)
                plt.imsave(os.path.join(destination,filename),image[0])
                counter += 1
                clear_output(wait=True)
                print("Transformation progress: "+str(counter)+"/"+str(len(filenames)))
            except:
                print('exception')
                pass



face_transform('/content/drive/MyDrive/120franky','/content/drive/MyDrive/FACEB',encoder_a,decoder_b)





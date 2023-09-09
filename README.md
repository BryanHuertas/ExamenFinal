# ExamenFinal
Examen final tratamiento de datos

Lo primero que se hace es instalar las librerias, posterior a ello tambien vamos a configurar el uso del gpu.

import tensorflow as tf
import os
import cv2
import imghdr
import imagesize

In [3]:
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
In [4]:
tf.config.list_physical_devices('GPU')
Out[4]:
[]
In [7]:
!pip install opencv-python

Posterior creamos una lista de extensiones permitidas, y listamos las imagenes.

data_dir = 'CarneDataset' 
image_exts = ['jpeg','jpg', 'bmp', 'png']
image_exts[3]
'png'


for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        print(image)

img=cv2.imread(os.path.join('CarneDataset','1train','train_02','14-CAPTURE_20220531_143820_956.PNG'))
In [152]:img.shape
Out[152]:(216, 384, 3)
In [153]:plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
Out[153]:<matplotlib.image.AxesImage at 0x26c8e14fd90>


A continuación cargamos nuestros datos, utilizaremos tensorflow para ello ya abremos importado numpy a continuacion, tambien podemos cargar otro lote de datos, y tambien vamos a identificar las mismas.

import numpy as np
In [191]:data = tf.keras.utils.image_dataset_from_directory('CarneDataset')
Found 2443 files belonging to 2 classes.
In [192]:data_iterator = data.as_numpy_iterator()
In [193]:batch = data_iterator.next()
In [194]:batch[1]
Out[194]:array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 0, 1, 1, 0, 0, 0, 1, 1])
In [195]:fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

Vamos a procesar los datos y darles una escala y dividir los datos.

data = data.map(lambda x,y: (x/255, y))
In [197]:scaled_iterator=data.as_numpy_iterator()
In [198]:batch = scaled_iterator.next()
In [199]:batch[0].min()
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1
In [203]:test_size+val_size+train_size

A continuacion construiremos una red neuronal, luego la entrenaremos y vamos a trazar nuestro rendimiento.
Red Neuronal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
In [207]:
model = Sequential()
In [208]:
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
In [209]:
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
In [210]:
model.summary()
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_3 (Conv2D)           (None, 254, 254, 16)      448       
                                                                 
 max_pooling2d_3 (MaxPoolin  (None, 127, 127, 16)      0         
 g2D)                                                            
                                                                 
 conv2d_4 (Conv2D)           (None, 125, 125, 32)      4640      
                                                                 
 max_pooling2d_4 (MaxPoolin  (None, 62, 62, 32)        0         
 g2D)                                                            
                                                                 
 conv2d_5 (Conv2D)           (None, 60, 60, 16)        4624      
                                                                 
 max_pooling2d_5 (MaxPoolin  (None, 30, 30, 16)        0         
 g2D)                                                            
                                                                 
 flatten_1 (Flatten)         (None, 14400)             0         
                                                                 
 dense_2 (Dense)             (None, 256)               3686656   
                                                                 
 dense_3 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 3696625 (14.10 MB)
Trainable params: 3696625 (14.10 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

Entrenamiento

logdir='logs'
In [212]:tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
In [213]:hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

Medicion de rendimiento

Perdida
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


PRECISION

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

EVALUACION 

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
In [217]:pre = Precision()
re = Recall()
acc = BinaryAccuracy()
In [218]:len (test)
Out[218]:
8
In [219]:for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(f'Presicion:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

PRUEBAS

yhat = model.predict(np.expand_dims(resize/255, 0))
1/1 [==============================] - 0s 23ms/step
In [284]:
yhat
Out[284]:
array([[0.9987762]], dtype=float32)
In [285]:
if yhat > 0.5: 
    print(f'Predicted class is carne')
else:
    print(f'Predicted class is hueso')








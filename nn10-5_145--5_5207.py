import numpy as np
import scipy.special
import winsound
import math
import matplotlib.pyplot as plt
#plt.use("Agg")
import sklearn
from sklearn import preprocessing
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import time
import tensorflow as tf
import random


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


random.seed(5)

# задаем для воспроизводимости результатов
#np.random.seed(2)
start_time = time.time()

dataset0 = np.loadtxt(r'D:\ITS\newfile2.csv', delimiter=',')
print(dataset0.shape)

X0 = dataset0[:209,0:81920]
Y0 = dataset0[:209,81920]

X1 = dataset0[235:,0:81920]
Y1 = dataset0[235:,81920]

#X0 = dataset0[:7077,0:1024]
#Y0 = dataset0[:7077,1024]

#X1 = dataset0[7077:8846,0:1024]
#Y1 = dataset0[7077:8846,1024]

#(X_train,y_train),(X_test,y_test)

X_train = X0
y_train = Y0

X_test = X1
y_test =Y1


X_val = dataset0[209:235,0:81920]
Y_val = dataset0[209:235,81920]


epochs = 1000
chanDim = -1
batch_size,img_rows,img_cols =4,256,320
X_train = X_train.reshape(X_train.shape[0],img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
X_val = X_val.reshape(X_val.shape[0],img_rows,img_cols,1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_val = X_val.astype("float32")
X_train /= 255
X_test /= 255
X_val /= 255

#X_train = (X_train0/255.0 * 0.99) + 0.01
#X_test = (X_test0/255.0 * 0.99) + 0.01

Y_train = np_utils.to_categorical(y_train,6)
Y_test =np_utils.to_categorical(y_test,6)
Y_val = np_utils.to_categorical(Y_val,6)

#for i in range(0, len(Y_test)):
#    print(y_test[i],'   ',Y_test[i])

model = Sequential()

model.add(Convolution2D(32,(5,5), padding="same",input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Convolution2D(32, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Convolution2D(64, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Convolution2D(64, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
#model.add(Convolution2D(128, (3, 3), padding="same"))
#model.add(Activation("relu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
#model.add(Dense(512))
#model.add(Activation("relu"))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(6))
model.add(Activation(tf.nn.softmax))

model.compile(loss ="categorical_crossentropy",optimizer="adam",metrics =["accuracy"])
H = model.fit(X_train,Y_train,callbacks =[ModelCheckpoint("digital-17072020-2.h5",monitor="val_acc",save_best_only=True, save_weights_only=False, mode="auto")],batch_size=batch_size,epochs=epochs,verbose=1,validation_data =(X_val,Y_val), shuffle=True)
score = model.evaluate(X_test,Y_test,verbose=0)
print("Test score: %f" % score[0])
print("Test accuracy (score[1): %f" % score[1])


plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('digital-17072020-2.png')
plt.show()
#plt.savefig(args["plot"])


duration = 1000  # millisecond
freq = 440  # Hz
winsound.Beep(freq, duration)


print("--- %s seconds ---" % (time.time() - start_time))

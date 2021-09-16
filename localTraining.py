import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

# style your matplotlib
mpl.style.use("seaborn-darkgrid")
# run this block

from tqdm import tqdm

files=os.listdir("dataset")
print(files)

image_array=[]  # it's a list later i will convert it to array
label_array=[]

path="dataset/"
# loop through each sub-folder in train
for i in range(len(files)):
    # files in sub-folder
    file_sub=os.listdir(path+files[i])

    for k in tqdm(range(len(file_sub))):
        try:
            img=cv2.imread(path+files[i]+"/"+file_sub[k])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(64,64))
            image_array.append(img)
            label_array.append(i)
        except:
            pass


import gc
gc.collect()       


#scaling of image array and convert to num array
image_array=np.array(image_array)/255.0
label_array=np.array(label_array)

#split image arra_array and label_array into train test array
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(image_array,label_array,test_size=0.15)


from tensorflow.keras import layers,callbacks,utils,applications,optimizers
from tensorflow.keras.models import Sequential,Model,load_model


model=Sequential()
# MobileNetV2 as an pretrained model 
pretrained_model=tf.keras.applications.EfficientNetB0(input_shape=(96,96,3),include_top=False, weights="imagenet")
model.add(pretrained_model)
model.add(layers.GlobalAveragePooling2D())
# add dropout to increase accuracy by not overfitting
model.add(layers.Dropout(0.3))
# add dense layer as final output
model.add(layers.Dense(1))
model.build(input_shape=(None,96,96,3))
model.summary()

model.compile(optimizer="adam",loss="mean_squared_error",metrics=["mae"])
# creating a chechpoint to save model at best accuarcy


#checks checkpoint to save model
ckp_path="trained_model/model"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
    monitor="val_mae",
    mode="auto",
    save_best_only=True,
    save_weights_only=True)

#reduce learning rate when val_mae doest not improve
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(factor=0.9,monitor="val_mae",
    mode="auto",cooldown=0,
    patience=5,verbose=1,min_lr=1e-6)

EPOCHS=200
BATCH_SIZE=64

history=model.fit(X_train,
    Y_train,
    validation_data=(X_test,Y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[model_checkpoint,reduce_lr]) 

#load best model
model.load_weights(ckp_path)   

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f: f.write(tflite_model)

#see the result
prediction_val=model.predict(X_test,batch_size=BATCH_SIZE)

#predict value
print(prediction_val[:20])
#original label
print(Y_test[:20])        
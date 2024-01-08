import cv2
import os
import PIL
#import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Sequential

image_directory = "brain_tumor_dataset/"
dataset = []
label = []

no_tumor_images = os.listdir(image_directory+ "no/")
yes_tumor_images = os.listdir(image_directory+ "yes/")

for i,image_name in enumerate(no_tumor_images):
    if image_name.split(".")[1] == "jpg":
        image = cv2.imread(image_directory+"no/"+image_name)
        image = Image.fromarray(image,"RGB")
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)
        
for j,image_names in enumerate(yes_tumor_images):
    if image_names.split(".")[1] == "jpg":
        images = cv2.imread(image_directory+"yes/"+image_names)
        images = Image.fromarray(images,"RGB")
        images = images.resize((64,64))
        dataset.append(np.array(images))
        label.append(1)
        
dataset = np.array(dataset)
label = np.array(label)

x_train,x_test,y_train,y_test = train_test_split(dataset,label,test_size=0.2)

x_train = normalize(x_train,axis=1)
x_test = normalize(x_test,axis=1)

#model
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation="relu",kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation="relu",kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=16,verbose=1,
          epochs=10,validation_data=(x_test,y_test),shuffle=False)

model.save("brain-tumors.h5")
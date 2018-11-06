import cv2
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
import os

if os.path.exists( r"C:\Users\Guilherme Bortolaso\Desktop\Skoods\model.h5"): ##To delete a pre-existing model
  os.remove(r"C:\Users\Guilherme Bortolaso\Desktop\Skoods\model.h5")
else:
  print("The file does not exist")

data_folder = r"C:\Users\Guilherme Bortolaso\Desktop\Skoods\Data\images/" ### Here is your images folder 
data = pd.read_csv(r"C:\Users\Guilherme Bortolaso\Desktop\Skoods\Data\airsim_rec.txt", sep="\t") ### Here is your data log 
steering = data["Steering"]
img_files = data["ImageFile"]

images=[]
for image in data["ImageFile"]:
    transition_image = cv2.imread(data_folder + image)
    #transition_image = transition_image[60:143,0:255] ##To crop sky 
    images.append(transition_image)
images=np.array(images)

## Creating DataSet
X_train=images
y_train=np.array(steering)

######Model Start Here#################################################
model=Sequential()
model.add(Lambda(lambda x:x/255 -0.5, input_shape=(144,256,3)))
#####Nvidea AutoPilot ConvNet ###############################################
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))


model.compile(loss="mse",optimizer="adam") #Adam Opmizaer, no Learning rate set mannualy

#Training set splitted in a rate of 20% for validation
historic=model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=10,verbose=1)

print(historic.history.keys())

"""
plt.plot(historic.history['loss'])
plt.plot(historic.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()"""

model.save("model.h5")



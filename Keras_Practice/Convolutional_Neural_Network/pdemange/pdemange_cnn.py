from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
import os
import numpy as np
import pickle

def outputOrganizer(x):
	finder = {
		'1':1,
		'2':2,
		'3':3,
		'4':4,
		'5':5,
		'6':6,
		'7':7,
		'8':8,
		'9':9,
		'0':10,
		'a':11,
		'b':12,
		'c':13,
		'd':14,
		'e':15,
		'f':16,
		'g':17,
		'h':18,
		'i':19,
		'j':20,
		'k':21,
		'l':22,
		'm':23,
		'n':24,
		'o':25,
		'p':26,
		'q':27,
		'r':28,
		's':29,
		't':30,
		'u':31,
		'v':32,
		'w':33,
		'x':34,
		'y':35,
		'z':36
	}
	return finder.get(x)

def createModel(): 
	model = Sequential()
	#Create the Convolutional Model
	model.add(Conv2D(32, (3,3), input_shape(150,150,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#Flatten and connect it to a regular Dense Neural Network
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(5, activation='relu'))
	adamOpt = optimizers.Adam(lr=.01)
	model.compile(loss='binary_crossentropy', optimizer=adamOpt, metrics=['accuracy'])
	return model

#Load the formatted data
def loadData(xFile, yFile):
	fX = open(xFile, 'r')
	data_x = pickle.load(fX)
	fX.close()

	fY = open(yFile, 'r')
	data_y = pickle.load(fY)
	fY.close()

	return data_x, data_y
	
#Do real quick formatting of the images for both testing and training!
def formatData(cDirect, fName): 
	data_x = []
	data_y = []
#	with open('training_x.txt', 'w+') as f:
	counter = 0
	for root, dirs, files in os.walk(cDirect): 
		print(files[0])
		for file in files:
			name = file[:file.index('.')]
			yName = np.array([])
			immy = load_img(os.path.join(root,file))
			immy = img_to_array(immy)
			data_x.append(immy)
			for char in name:
				yName = np.append(yName, outputOrganizer(char))
			data_y.append(yName)
			counter += 1
	print('Length of X data: '+str(len(data_x)))
	print('Length of Y data: '+str(len(data_y)))
	fx = open(fName+'_x.txt', 'w+')
	pickle.dump(np.asarray(data_x), fx)
	fx.close()
	fy = open(fName+'_y.txt', 'w+')
	pickle.dump(np.asarray(data_y), fy)
	fy.close()



from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from keras import optimizers
import os
import numpy as np
import pickle


def createModel(): 
	model = Sequential()
	#Create the Convolutional Model
	model.add(Conv2D(32, (3,3), input_shape=(50,200,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	#Flatten and connect it to a regular Dense Neural Network
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(35, activation='softmax'))
	model.add(Dens)
	adamOpt = optimizers.Adam(lr=.01)
	model.compile(loss='binary_crossentropy', optimizer=adamOpt, metrics=['accuracy'])
	return model

class dataHandler:
	def __init__(self):
		self.values = '1234567890abcdefghijklmnopqrstuvwxyz'
		self.bList = self.genOutList()
		print(self.bList)

	def genOutList(self):
		tempy = [[0]]*36
		for x in range(0,36):
			temp = [0]*36
			temp[x] = 1
			tempy[x] = temp
		return tempy

	def lookup(self, x):
		return self.bList[self.values.index(x)]

	def reverse(self, x):
		return self.values[bList.index(x)]

	def loadData(xFile, yFile):
		fX = open(xFile, 'rb')
		data_x = pickle.load(fX)
		fX.close()

		fY = open(yFile, 'rb')
		data_y = pickle.load(fY)
		fY.close()

		return data_x, data_y

#Load the formatted data

	
#Do real quick formatting of the images for both testing and training!
def formatData(cDirect, fName, dataOrg): 
	data_x = []
	data_y = []
#	with open('training_x.txt', 'w+') as f:
	counter = 0
	for root, dirs, files in os.walk(cDirect): 
		for file in files:
			name = file[:file.index('.')]
			yName = []
			immy = load_img(os.path.join(root,file))
			immy = img_to_array(immy)
			data_x.append(immy)
			for char in name:
				yName.append(dataOrg.lookup(char))
			data_y.append(yName)
			counter += 1
	print('Length of X data: '+str(len(data_x)))
	print('Length of Y data: '+str(len(data_y)))
	fx = open(fName+'_x.txt', 'wb+')
	pickle.dump(np.asarray(data_x), fx)
	fx.close()
	fy = open(fName+'_y.txt', 'wb+')
	pickle.dump(np.asarray(data_y), fy)
	fy.close()

def evaluateModel(network, xTest, yTest):
	score = network.evaluate(xTest, np.asarray(yTest))
	print('Testing Accuracy - %s : %.f2%%' % (network.metrics_names[1], score[1]*100))

if __name__ == '__main__': 
	c = dataFinder()
	formatData('train/training','training',c)
	formatData('test/testing','testing',c)
	f = open('training_y.txt','rb')
	yTrain = pickle.load(f)
	f.close()
	print(yTrain)
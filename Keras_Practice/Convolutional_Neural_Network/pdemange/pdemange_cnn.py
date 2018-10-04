from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras.models import Model
from keras.layers import Embedding,Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from keras.utils.np_utils import to_categorical
from keras import optimizers
import os
import numpy as np
import h5py


def createModel(): 
	#Create the Convolutional Model
	inp = Input(shape=(50,200,3))
	x = Conv2D(32, (3,3), input_shape=(50,200,3), activation='relu')(inp)
	x = MaxPooling2D(pool_size=(2,2))(x)
 	x = Conv2D(32, (3,3), activation='relu')(x)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = Conv2D(64, (3,3), activation='relu')(x)
	x = MaxPooling2D(pool_size=(2,2))(x)
	#Flatten and connect it to a regular Dense Neural Network
	x = Flatten()(x)
	x = Dense(64, activation='relu')(x)
	x = Dropout(0.5)(x)
	preout1 = Dense(32, activation='relu')(x)
	out1 = Dense(5*36, activation='softmax')(preout1)
	model = Model(inputs=inp, outputs=out1)
	adamOpt = optimizers.Adam(lr=.01)
	model.compile(loss='binary_crossentropy', optimizer=adamOpt, metrics=['accuracy'])
	return model

class dataHandler:
	def __init__(self):
		self.values = '1234567890abcdefghijklmnopqrstuvwxyz'
		self.bList = self.genOutList()
		print(self.bList)

	def genOutList(self):
		tempy = np.array([[0]*36]*36)
		print(tempy)
		for x in range(0,36):
			temp = np.zeros(36)
			temp[x] = 1
			tempy[x] = temp
		return tempy

	def lookup(self, x):
		return self.bList[self.values.index(x)]

	def reverse(self, x):
		return self.values[bList.index(x)]

	def loadData(self, xFile, yFile, alias):
		fx = h5py.File(xFile, 'r')
		data_x = fx[alias+'_x'][:]
		fx.close()

		fy = h5py.File(yFile, 'r')
		data_y = fy[alias+'_y'][:]
		fy.close()

		return data_x, data_y

#Load the formatted data

	
#Do real quick formatting of the images for both testing and training!
def formatData(cDirect, fName, dataOrg): 
	data_x = []
	data_y = []
	counter = 0
	for root, dirs, files in os.walk(cDirect): 
		for file in files:
			name = file[:file.index('.')]
			yName = np.array([])
			immy = load_img(os.path.join(root,file))
			immy = img_to_array(immy)
			data_x.append(immy)
			for char in name:
				yName = np.append(yName, dataOrg.lookup(char))
			data_y.append(yName)
			counter += 1
	print(data_y)
	#print('Length of X data: '+str(len(data_x)))
	#print('Length of Y data: '+str(len(data_y)))
	hf = h5py.File(fName+'_x.h5', 'w')
	hf.create_dataset(fName+'_x', data=np.array(data_x))
	hf.close()
	hf = h5py.File(fName+'_y.h5', 'w')
	hf.create_dataset(fName+'_y', data=np.array(data_y))
	hf.close()

def evaluateModel(network, xTest, yTest):
	score = network.evaluate(xTest, yTest)
	print('Testing Accuracy - %s : %.2f%%' % (network.metrics_names[1], score[1]*100))

if __name__ == '__main__': 
	c = dataHandler()
	#print(c.lookup('1'))
	formatData('train/training','training',c)
	formatData('test/testing','testing',c)
	xTrain, yTrain = c.loadData('training_x.h5', 'training_y.h5','training')
	xTest, yTest = c.loadData('testing_x.h5', 'testing_y.h5','testing')
	#print(type(yTrain))
	#print(type(yTest))
	net = createModel()
	net.fit(xTrain, yTrain, epochs=20, batch_size=16)
	evaluateModel(net, xTest, yTest)
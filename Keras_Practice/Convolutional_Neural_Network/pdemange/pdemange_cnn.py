from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras.preprocessing.text import Tokenizer
from keras.models import Model, model_from_json
from keras.layers import Embedding,Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from keras.utils.np_utils import to_categorical
from keras import optimizers
import os
import numpy as np
import h5py
from PIL import Image
from time import sleep

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
		#print(self.bList)

	def genOutList(self):
		tempy = np.array([[0]*36]*36)
		#print(tempy)
		for x in range(0,36):
			temp = np.zeros(36)
			temp[x] = 1
			tempy[x] = temp
		return tempy

	def lookup(self, x):
		return self.bList[self.values.index(x)]

	def reverse(self, x):
		idx = np.argwhere(x==1)
		print(idx)
		return self.values[idx[0][0]]

	def answer(self, resp): 
		y = 0
		ans = ''
		for x in range(36, 226, 36):
			if y == 180:
				break
			temp = resp[y:x]
			print('Looking for: '+str(temp))
			ans += self.reverse(temp)
			y = x
		return ans

class DataHandler:
	def __init__(self, allChars):
		self.tok = self.createTokenizer(allChars)

	def createTokenizer(self, chars):
		temp = Tokenizer(char_level=True)
		temp.fit_on_texts(chars)
		return temp 

	def toCategorical(self, text):
		return to_categorical(self.tok.texts_to_sequences(text))

	def fromCategorical(self, cat): 
		temp = np.zeros(shape=(1,len(cat)))
		for i in range(cat.shape[0]):
			temp[0][i] = np.argmax(cat[i])
		return self.tok.sequences_to_texts(temp)

	def loadData(self, xFile, yFile, alias):
		fx = h5py.File(xFile, 'r')
		data_x = fx[alias+'_x'][:]
		fx.close()

		fy = h5py.File(yFile, 'r')
		data_y = fy[alias+'_y'][:]
		fy.close()

		return data_x, data_y



#Load the formatted data

def loadNetwork(nFile):
	with open(nFile+'_options.json', 'r') as f:
		model = model_from_json(f.read())
	model.load_weights(nFile+'_weights.h5')
	adamOpt = optimizers.Adam(lr=.01)
	model.compile(loss='binary_crossentropy', optimizer=adamOpt, metrics=['accuracy'])
	return model
	
def saveNetwork(nNet, nFile):
	with open(nFile+'_options.json', 'w') as f:
		f.write(nNet.to_json())
	nNet.save_weights(nFile+'_weights.h5')
	print('Model saved!')

def coolPredict(nNet, lookupHandler, vPredict, vAnswer=None):
	val = Image.fromarray(vPredict[0], 'RGB')
	val.show()
	sleep(5)
	ans = nNet.predict(vPredict)
	tAns = lookupHandler.answer(ans[0])
	if vAnswer is not None:
		temp = lookupHandler.answer(vAnswer)
		if tAns == temp:
			print(tAns+' == '+temp)
		else:
			print(tAns+' != '+temp)
	input('Press enter to continue...')

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
	c = DataHandler('1234567890abcdefghijklmnopqrstuvwxyz')
	tt = c.toCategorical('1234567890abcdefghijklmnopqrstuvwxyz')
	print(tt)
	print(c.fromCategorical(tt))
	#print(c.lookup('1'))
	#formatData('train/training','training',c)
	#formatData('test/testing','testing',c)
	#xTrain, yTrain = c.loadData('training_x.h5', 'training_y.h5','training')
	#xTest, yTest = c.loadData('testing_x.h5', 'testing_y.h5','testing')
	#net = loadNetwork('captcha')
	#print(type(yTrain))
	#print(type(yTest))
	#net = createModel()
	#for x_data, y_data in zip(xTrain, yTrain):
	#net.fit(np.array([x_data]), np.array([y_data]), epochs=20)
	#evaluateModel(net, xTest, yTest)
	#print(net.predict(np.array([xTest[0]])))
	#for inc in range(len(xTest)):
#		coolPredict(net, c, np.array([xTest[inc]]), yTest[inc])
	#saveNetwork(net,'captcha')
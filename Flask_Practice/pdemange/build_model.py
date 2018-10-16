from keras.applications import vgg16
from keras.layers import Dense
from keras.models import Sequential

def buildModel():
	baseModel = vgg16.VGG16()
	baseModel.layers.pop()
	model = Sequential()
	for m in baseModel.layers:
		model.add(m)
	for layer in model.layers:
		layer.trainable=False
	model.add(Dense(2, activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.optimizer.lr=.0001
	return model

from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras import optimizers

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

#Do real quick formatting of the images for both testing and training!
def formatData(): 
	trainDataGen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	testDataGen = ImageDataGenerator(rescale=1./255)

	trainGenerator = trainDataGen.flow_from_directory(
		'training',
		target_size=(150,150),
		batch_size=16, 
		class_mode='binary')

	testGenerator = testDataGen.flow_from_directory(
		'testing',
		target_size=(150,150),
		batch_size=16,
		class_mode='binary')

formatData()
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

""" code based from a Keras Blog
	(https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
	and the repository that I got the code from is
	(https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
"""

"""Citation for database:
	Folder [test-multiple_fruits](test-multiple_fruits) contains images with multiple fruits. Some of them are partially covered by other fruits. Also, they were captured in different lighting conditions compared to the fruits from Training and Test folder. This is an excelent test for real-world detection.
	"""

# dimensions of our images
img_width, img_height = 150, 150

# locations of the data for training and testing
train_dir = 'fruit_data/Training'
test_dir = 'fruit_data/Test'

train_sample = 2000
test_sample = 800
epochs = 2
batch_size = 16
classes = 3

# formating shape in this order because we are using tenserflow
input_shape = (img_width, img_height, 3)

def create_model():
	neural_network = Sequential()
	neural_network.add(Conv2D(32, (3, 3), input_shape=input_shape))
	neural_network.add(Activation('relu'))
	neural_network.add(MaxPooling2D(pool_size=(2, 2)))

	neural_network.add(Conv2D(32, (3, 3)))
	neural_network.add(Activation('relu'))
	neural_network.add(MaxPooling2D(pool_size=(2, 2)))

	neural_network.add(Conv2D(64, (3, 3)))
	neural_network.add(Activation('relu'))
	neural_network.add(MaxPooling2D(pool_size=(2, 2)))

	neural_network.add(Flatten())
	neural_network.add(Dense(64))
	neural_network.add(Activation('relu'))
	neural_network.add(Dropout(0.5))
	neural_network.add(Dense(classes))
	neural_network.add(Activation('softmax'))
	neural_network.compile(loss='binary_crossentropy',
              	  optimizer='rmsprop',
				  metrics=['accuracy'])
	return neural_network

def training_set():
	# modifies images to keep the neural network from overfitting
	train_augment = ImageDataGenerator(
	    rescale=1. / 255,
	    shear_range=0.2,
	    zoom_range=0.2,
	horizontal_flip=True)

	# this will also modify the images for testing, but only scaling
	test_augment = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_augment.flow_from_directory(
		train_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')

	test_generator = test_augment.flow_from_directory(
		test_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')
	return train_generator, test_generator

neural_network = create_model()
train_generator, test_generator = training_set()

# training our network and then testing our neural network
neural_network.fit_generator(
	train_generator,
	steps_per_epoch=train_sample // batch_size,
	epochs=epochs,
#	validation_data=test_generator,
#	validation_steps=test_sample // batch_size,
	verbose=2)

# how to modify image
trial_augment = ImageDataGenerator(rescale=1. / 255)
# creating a manual test
trial_gen = trial_augment.flow_from_directory(
                'test_data',
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode='categorical')

# evaluating using my own dataset
trial_result = neural_network.predict_generator(trial_gen, verbose=1)
print(trial_result)

# saving our trained neural network
model.save_weights('first_try.h5')

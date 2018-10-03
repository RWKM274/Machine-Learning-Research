from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K, optimizers
import os

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
train_dir = 'additional_testing'
test_dir = 'fruit_data/Test'

# name of neural network model
neural_network_name = 'neural_network.h5'

train_sample = 2000
test_sample = 800
epochs = 4
batch_size = 16
classes = 3

# formating shape in this order because we are using tenserflow
input_shape = (img_width, img_height, 3)

# for debugging purposes
Debug = False

# to train or not to train
load_network = False

# Do you want to test your neural network using the dataset's test dir
default_dataset_test = False

# Do you want to test using custom neural network
custom_test = False

# Use evaluate function
evaluate_test = False

""" creates a model using conv2D, an activation of relu, and a maxpooling and
	there are three of thoses. After that it puts the 3 Dimentional array and
	condenses it into a 1 Dimentional array. Using that, it will do a softmax
	which is good for classification with three or more options. Then
	we compile the model using binary crossentropy and rmsprop optimizer."""
def create_model():
	optimization_rate = optimizers.rmsprop(lr=0.1)
	neural_network = Sequential()
	neural_network.add(Conv2D(32, (3, 3), input_shape=input_shape))
	neural_network.add(Activation('relu'))
	neural_network.add(MaxPooling2D(pool_size=(2, 2)))
	neural_network.add(Dropout(0.5))

	neural_network.add(Conv2D(32, (3, 3)))
	neural_network.add(Activation('relu'))
	neural_network.add(MaxPooling2D(pool_size=(2, 2)))

	neural_network.add(Conv2D(64, (3, 3)))
	neural_network.add(Activation('relu'))
	neural_network.add(MaxPooling2D(pool_size=(2, 2)))
	neural_network.add(Dropout(0.5))

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

def augment_training_set():
	# modifies images to keep the neural network from overfitting
	train_augment = ImageDataGenerator(
	    rescale=1. / 255,
	    shear_range=0.2,
	    zoom_range=0.2,
		horizontal_flip=True)

	# this will also modify the images for testing, but only scaling
	test_augment = ImageDataGenerator(rescale=1. / 255)

	""" modifies the images to a particular width and height
		(target_size) within the train directory(train_dir).
		And identifies each of the group based on their
		respective directory names."""
	train_generator = train_augment.flow_from_directory(
		train_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')

	""" modifies the images to a particular width and height(target_size)
		within the test directory(dir). And classifies using the names of
		the directory names inside the test diretory."""
	test_generator = test_augment.flow_from_directory(
		test_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')
	return train_generator, test_generator

def print_accuracy(num_correct, num_total_files):
	print('Accuracy: ' + str(num_correct/num_total_files * 100) + '%')

def testing_neural_network(neural_network):
	array_of_items = ['Apple', 'Banana', 'Pear']
	file_spot = 0
	num_total_files = 0
	num_correct = 0
	for dir_name in array_of_items:
		for root, dirs, files in os.walk('./' + test_dir + '/' + dir_name):
			num_total_files += len(files)
			if(Debug):
				print('-'*80)
				print(len(files))
				print(dir_name)
				print('-'*80)
			for pics in files:
				file_name = test_dir + '/' + dir_name + '/' + pics
				test_img = image.load_img(file_name, target_size=(img_width, img_height))
				x = image.img_to_array(test_img)
				inputx = x.reshape([-1, img_width, img_height, 3])
				trailx = neural_network.predict(inputx, verbose=1)
				if(Debug):
					print('*'*80)
					print(str(trailx[0]) + ' is the array')
					print('*'*80)
				if(trailx[0][file_spot] == 1):
					num_correct += 1

		# to check the array location
		file_spot += 1
	print_accuracy(num_correct, num_total_files)
	if(Debug):
		print(str(num_total_files) + ' total files')
		print(str(num_correct) + ' correct ones')


# training our network and then testing our neural network
def training_neural_network(neural_network, train_generator, test_generator):
        neural_network.fit_generator(
                train_generator,
                steps_per_epoch=train_sample // batch_size,
                epochs=epochs,
#               validation_data=test_generator,
#               validation_steps=test_sample // batch_size,
                verbose=2)

		
def evaluate_custom_images():
	 # testing the neural network with customized images
	list_of_options = ['apple', 'banana', 'pear']
	num_total_files = 0
	num_correct = 0
	for root, dirs, files in os.walk('./test_data'):
		for pics in files:
			num_total_files += 1
			file_name = 'test_data/' + pics
			test_img = image.load_img(file_name, target_size=(img_width, img_height))
			x = image.img_to_array(test_img)
			inputx = x.reshape([-1, img_width, img_height, 3])
			result_neural_network = neural_network.predict(inputx, verbose=1)

			if(Debug):
				print('picture name: ' + pics)

			""" grabs the fruit name from the name of the file. For
				example, banana_2.jpg => banana"""

			word_of_pic = pics.split('_')[0]
			location_of_word_in_array = list_of_options.index(word_of_pic)

			""" increments the amount of times the neural network
				guessed correctly if it returns 1 for the correct
				category. WARNING: if more than one category is
				flaged, this code will not catch it."""
			if(result_neural_network[0][location_of_word_in_array] == 1):
				num_correct += 1
				if(Debug):
					print('The neural network guessed correctly')
			else:
				if(Debug):
					print('The neural network guessed incorrect')

	if(Debug):
		print(num_correct)
	print(str(num_total_files) + ' total')
	print('Accuracy: ' + str(num_correct/num_total_files * 100) + '%')


def evaluate_use_evaluate_function():
	test_augment = ImageDataGenerator(rescale=1. / 255)
	testing_generator = test_augment.flow_from_directory(
		test_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')
	results = neural_network.evaluate_generator(testing_generator)
	print('Loss: ' + str(results[0]) + '\n' + 'Accuracy: ' + str(results[1]*100) + '%')



# Creating and Traing our Convolutional Neural Network

# Creating the model
if(load_network == False):
	neural_network = create_model()

# Returns the modification of the training set
if(load_network == False):
	train_generator, test_generator = augment_training_set()

# Trains the Neural Network
if(load_network == False):
	training_neural_network(neural_network, train_generator, test_generator)


# loading an existing Neural Network model
if(load_network):
	try:
		neural_network = load_model(neural_network_name)
	except:
		print('ERROR: could not load ' + neural_network_name)




""" this will check to see if the Neural Network with the Test directory provided
	with the dataset. It will display the Accuracy of the Neural Network
"""
if(default_dataset_test):
	testing_neural_network(neural_network)


if(custom_test):
	evaluate_custom_images()

if(evaluate_test):
	evaluate_use_evaluate_function()
	


# how to modify image
# trial_augment = ImageDataGenerator(rescale=1. / 255)
# creating a manual test
# trial_gen = trial_augment.flow_from_directory(
     #   	'test_data',
    #            target_size=(img_width, img_height),
   #             batch_size=batch_size,
  #              class_mode='categorical')

# evaluating using my own dataset
# trial_result = neural_network.predict_generator(trial_gen, verbose=1)





# test_img = image.load_img('test_data/banana_original.jpg', target_size=(img_width, img_height))

# x = image.img_to_array(test_img)
# inputpy = x.reshape([-1, img_width, img_height, 3])

# test2 = image.load_img('test_data/original_pear.jpg', target_size=(img_width, img_height))

# n = image.img_to_array(test2)
# input2 = n.reshape([-1, img_width, img_height, 3])

# trial2 = neural_network.predict(input2, verbose=1)
# trial_result = neural_network.predict(inputpy, verbose=1)
# # print('banana: ' + trial_result)
# # print('pear: ' + trial2)
# print(trial_result)
# print(trial2)
# print(type(trial_result))

# ending = neural_network.evaluate_generator(test_generator, verbose=1)
# print(ending)

# saving our trained neural network
if(load_network == False):
	neural_network.save(neural_network_name)

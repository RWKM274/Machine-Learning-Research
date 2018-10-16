'''
counter = 0
x_num = 0
y_num = 0
for i in os.listdir(path_own_comp):
    #print(i)
    if(i.endswith('.jpg')):
        if(x_num + y_num == 2000):
            break
        image = load_img(path_own_comp + i, target_size = (150, 150))
        array_image = img_to_array(image)
        #print(array_image.shape)
        if(i.startswith('dog')):
            if(not x_num == 1000):
                training_input.append(array_image)
                training_output.append([1, 0])
                x_num += 1
        if(i.startswith('cat')):
            if(not y_num == 1000):
                training_input.append(array_image)
                training_output.append([0, 1])
                y_num += 1
        counter = counter+1
        precentage = (counter / 25000) * 100
        if(precentage % 10 == 0):
            print(precentage, "%")


print("Data cleaning finish")

training_input = np.array(training_input)
training_output = np.array(training_output)


print(type(training_input))
print(training_input[1])

with h5py.File('cnn_object_input.hdf5', 'w') as f:
    write_in = f.create_dataset('input_data', data = training_input)
    write_out = f.create_dataset('output_data', data = training_output)


with h5py.File('cnn_object_input.hdf5', 'r') as f:
    training_input = f['input_data']
    training_output = f['output_data']
'''


from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import cv2
import h5py
import os



def model_create():
    
    model = Sequential()

    filters_1, filters_2, filters_3 = 32, 32, 64

    kernal_size = (3, 3)

    input_shape = (150, 150, 3)
    pool_size = (2, 2)


    model.add(Conv2D(filters_1, kernal_size, input_shape = (150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = pool_size))

    model.add(Conv2D(filters_2, kernal_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = pool_size))

    model.add(Conv2D(filters_3, kernal_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = pool_size))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])


    return model




def dataset_create(mother_folder_name, img_width, img_height):
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1. /255)

    train_generator = train_datagen.flow_from_directory(
        mother_folder_name,
        target_size = (img_width, img_height),
        class_mode = 'binary')

    return train_generator





if __name__ == "__main__":
    


    path_lab = "../../../dog_cat/train/input/"
    path_own_comp = "../../Desktop/dog_cat/train/"
    test_path_own = "../../Desktop/dog_cat/test1/"
    test_path_lab = "../../../dog_cat/test1/"
    
    M, N= 150, 150



    print("Start inputing data")

    model = model_create()

    x_train = dataset_create(path_lab, M, N)

    model.fit_generator(x_train, steps_per_epoch = len(x_train), epochs = 20)





    while(1 == 1):
        x = int(input("Please input a number between 1 - 12500, quit by input -1"))
        if(x == -1):
            break
        else:
            x = str(x) + ".jpg"
            image = load_img(test_path_lab + x, target_size = (150, 150))
            array_image = img_to_array(image)
            print(model.predict(arr_image))

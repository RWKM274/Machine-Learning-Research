import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras.applications import vgg16
from keras import regularizers


class AnimalClassifier:
    def __init__(self, trainDirectory, testDirectory, validDirectory, bSize=16):
        self.batchSize = bSize
        self.trainingNumber, self.testingNumber, self.validationNumber = self.countData(trainDirectory, testDirectory, validDirectory)
        self.train, self.test, self.validate = self.prepareData(trainDirectory, testDirectory, validDirectory)
        self.network = self.createModel()


    def countData(self, trD, teD, vD):
        trCount = len(os.listdir(trD+'/Cat'))+len(os.listdir(trD+'/Dog'))
        teCount = len(os.listdir(teD + '/Cat')) + len(os.listdir(teD + '/Dog'))
        vCount = len(os.listdir(vD + '/Cat')) + len(os.listdir(vD + '/Dog'))
        return trCount, teCount, vCount

    def prepareData(self, trD, teD, vD):

        testDatagen = ImageDataGenerator(rescale=.1/255)
        trainDatagen = ImageDataGenerator(rescale=.1/255, horizontal_flip=True, zoom_range=0.2,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2)
        train = trainDatagen.flow_from_directory(
            trD,
            target_size=(150, 150),
            batch_size=self.batchSize,
            class_mode='binary')

        test = testDatagen.flow_from_directory(
            teD,
            target_size=(150, 150),
            batch_size=self.batchSize,
            class_mode='binary')

        valid = testDatagen.flow_from_directory(
            vD,
            target_size=(150, 150),
            batch_size=self.batchSize,
            class_mode='binary')

        #print(str(train))
        return train, test, valid

    def createModel(self):
        vggModel = vgg16.VGG16(include_top=False)
        model = Sequential()
        for layer in vggModel.layers:
            layer.trainable=False
            model.add(layer)
        #Creating the convolutional layer
        '''
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(150,150,3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        
        
        
        model.optimizer.lr = .001
        '''
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))  # Dropout to prevent overfitting
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model

    def saveWeights(self, fileName):
        self.network.save_weights(fileName+'.h5')

    def loadWeights(self, weightFile):
        self.network.load_weights(weightFile+'.h5')

    def practice(self):
        #print(self.train)
        self.network.fit_generator(self.train, steps_per_epoch=self.trainingNumber/self.batchSize, validation_data=self.validate, validation_steps=self.validationNumber/self.batchSize,epochs=100)

    def evaluate(self):
        trainingScores = self.network.evaluate_generator(self.train, steps=self.trainingNumber/10)
        print('Training accuracy - %s : %.2f%%' % (self.network.metrics_names[1], trainingScores[1]*100))
        testingScores = self.network.evaluate_generator(self.test, steps=self.testingNumber/10)
        print('Testing accuracy - %s : %.2f%%' % (self.network.metrics_names[1], testingScores[1]*100))

    def predict(self, imgToPredict):
        imP = load_img(imgToPredict, target_size=(150,150))
        final = img_to_array(imP)
        return self.network.predict(final)

if __name__ == '__main__':
    net = AnimalClassifier('data/training','data/testing','data/validation',12)
    net.practice()
    #net.loadWeights('final2')
    net.evaluate()
    net.saveWeights('final2')
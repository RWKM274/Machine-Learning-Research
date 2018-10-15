import numpy as np
import os
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Flatten, Dropout, Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras.applications import inception_v3,vgg16
from keras import regularizers


class AnimalClassifier:
    def __init__(self, trainDirectory, testDirectory, validDirectory, bSize=16):
        self.batchSize = bSize
        self.trainingNumber, self.testingNumber, self.validationNumber = self.countData(trainDirectory, testDirectory, validDirectory)
        self.train, self.test, self.validate = self.prepareData(trainDirectory, testDirectory, validDirectory)
        self.network = self.createModel()


    def countData(self, trD, teD, vD):
        trCount = len(os.listdir(trD+'/cat'))+len(os.listdir(trD+'/dog'))
        teCount = len(os.listdir(teD + '/cat')) + len(os.listdir(teD + '/dog'))
        vCount = len(os.listdir(vD + '/cat')) + len(os.listdir(vD + '/dog'))
        return trCount, teCount, vCount

    def prepareData(self, trD, teD, vD):

        testDatagen = ImageDataGenerator()
        trainDatagen = ImageDataGenerator()
        train = trainDatagen.flow_from_directory(
            trD,
            target_size=(224, 224),
            batch_size=self.batchSize,
            classes=['cat','dog'])

        test = testDatagen.flow_from_directory(
            teD,
            target_size=(224, 224),
            batch_size=self.batchSize,
            classes=['cat','dog'])

        valid = testDatagen.flow_from_directory(
            vD,
            target_size=(224, 224),
            batch_size=self.batchSize,
            classes=['cat','dog'])

        #print(str(train))
        return train, test, valid

    def createModel(self):
        baseModel = vgg16.VGG16()
        baseModel.layers.pop()
        model = Sequential()
        for m in baseModel.layers:
            model.add(m)
        # add a global spatial average pooling layer
        # = baseModel.output
        #x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        #x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
       
        #model = Model(inputs=baseModel.input, outputs=predictions)
        #for layer in baseModel.layers:
        #    layer.trainable = False
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.optimizer.lr = .0001
        return model
     
    def saveWeights(self, fileName):
        self.network.save_weights(fileName+'.h5')

    def saveModel(self, name):
        with open(name+'.json', 'w') as f:
          f.write(self.network.to_json())
        self.network.save_weights(name+'_weights.h5')
    
    def loadWeights(self, weightFile):
        self.network.load_weights(weightFile+'.h5')

    def practice(self):
        #print(self.train)
        self.network.fit_generator(self.train, steps_per_epoch=self.trainingNumber/self.batchSize, validation_data=self.validate, validation_steps=self.validationNumber/self.batchSize,epochs=15)

    def evaluate(self):
        trainingScores = self.network.evaluate_generator(self.train, steps=self.trainingNumber/10)
        print('Training accuracy - %s : %.2f%%' % (self.network.metrics_names[1], trainingScores[1]*100))
        testingScores = self.network.evaluate_generator(self.test, steps=self.testingNumber/10)
        print('Testing accuracy - %s : %.2f%%' % (self.network.metrics_names[1], testingScores[1]*100))

    def predict(self, imgToPredict):
        imP = load_img(imgToPredict, target_size=(224,224))
        final = img_to_array(imP)
        return self.network.predict(final)

if __name__ == '__main__':
    net = AnimalClassifier('data/train','data/test','data/validate',24)
    #net.practice()
    net.loadWeights('origin_weights')
    net.evaluate()
    #net.saveWeights('origin')

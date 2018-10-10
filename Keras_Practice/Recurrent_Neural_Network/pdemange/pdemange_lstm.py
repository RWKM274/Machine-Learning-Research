from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
import collector

coll = collector.CaptionCollector()
caps = []
for files in os.listdir():
    if '.vtt' in files:
        caps += coll.readAllCaptions(files)
nCaps = coll.formatCaptions(caps, {'\n':'','D:':'\nD:', 'A:':'\nA:','Arin:': '\nA:', 'Dan:':'\nD:', '(Arin)':'\nA:', '(Danny)':'\nD:',
                                   'danny:':'\nD:', 'arin:':'\nA:', '[danny]':'\nD','[arin]':'\nA:', '[Danny]':'\nD','[Arin]':'\nA:','Danny':'\nD:'})
sentences = ' '.join(nCaps).lower() # ['Hello', 'world'] --> 'hello world'
chars = sorted(list(set(sentences))) # <-- get length of unique characters Ex. 30
charIndices = dict((c,i) for i, c in enumerate(chars))# <-- {a:1,b:2}
indicesChar = dict((i,c) for i, c in enumerate(chars))# <-- {1:a. 2:b}
maxl = 40

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def buildModel(maxLen, charLen):
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxLen, charLen))) # [[0..1], [0..0]] <-- one hot encode
    #model.add(LSTM(64))
    model.add(Dense(charLen, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.optimizer.lr=.01
    return model

def prepareData(maxLen, sentencesCom, charLen, steps=3):
    allSent = []
    nextChars = []
    for i in range(0, len(sentencesCom) - maxLen, steps):
        allSent.append(sentencesCom[i:i + maxLen])
        nextChars.append(sentencesCom[i + maxLen])

    x = np.zeros((len(allSent), maxLen, charLen), dtype=np.bool)
    y = np.zeros((len(allSent), charLen), dtype=np.bool)
    for i, sent in enumerate(allSent):
        for c, char in enumerate(sent):
            x[i, c, charIndices[char]] = 1
        y[i, charIndices[nextChars[i]]] = 1

    return x, y

x, y = prepareData(maxl, sentences, len(chars))

model = buildModel(maxl, len(chars))

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    if epoch+1 == 1 or (epoch+1) % 5 == 0:
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(sentences) - maxl - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = sentences[start_index: start_index + maxl]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxl, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, charIndices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indicesChar[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

model.fit(x,y,batch_size=512, epochs=50, verbose=1, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
model.save('first_try.h5')

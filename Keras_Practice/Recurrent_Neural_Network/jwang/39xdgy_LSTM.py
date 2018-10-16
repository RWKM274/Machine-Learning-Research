'''
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.utils import np_utils
import random, sys, io

# maxlen = how long each sentence is
# chars = how many unique characters in the whole text

global model, chars, dataX

def read_input(input_path):
    with open(input_path, 'r') as f:
        input_english = f.read()

    input_english = input_english.lower()
    #print(input_english)

    chars = sorted(list(set(input_english)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    #print(chars)
    #print(char_to_int)

    n_chars = len(input_english)
    n_vocab = len(chars)
    #print(n_chars)
    #print(n_vocab)
    

    return input_english, chars, char_to_int, n_chars, n_vocab




def create_model(seq_length, n_vocab):
    model = Sequential()
    model.add(LSTM(128, input_shape = (seq_length, 1), return_sequences = False))
    #model.add(Dropout(0.2))
#    model.add(LSTM(256))
    #model.add(Dropout(0.2))
    model.add(Dense(n_vocab, activation = 'softmax'))

    optimizer = RMSprop(lr = 0.01)

    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    return model

def gene_text():
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

    

    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x/float(n_vocab)
        prediction = model.predict(x, verbose = 0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]



def sample(preds, temperature = 1.0):
    preds = np.asarry(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

        
def on_epoch_end(epoch, _):
    print()
    print('-----Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(raw_text) - seq_length - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = raw_text[start_index: start_index + seq_length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')

        sys.stdout.write(generated)
        char_indices = dict((c, i) for i, c in enumerate(chars))
        for i in range(100):
            x_pred = np.zeros((1, seq_length, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1
            preds = model.predict(x_pred, diversity)
            next_index = sample(x_pred, diversity)
            next_char = char_to_int[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    
raw_text, chars, char_to_int, n_chars, n_vocab = read_input("./11-0.txt")

#print(raw_text, "\n", chars, "\n", char_to_int, "\n", n_chars, "\n", n_vocab)


seq_length = 100
dataX, dataY = [], []


for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print(len(dataX[0]))



for i in range(0, n_patterns):
    for j in range(0, seq_length):
        test = np.zeros(n_vocab)
        test[dataX[i][j]] = 1
        dataX[i][j] = test
#print(dataX[0])



'''
#print(n_patterns)
#print(dataX[0][101])
#print(dataX)
X = np.zeros((seq_length, n_vocab))


for i in range(0, seq_length):
    for j in range(0, n_vocab):
        binary_list = np.zeros(n_vocab)
        print(dataX[seq_length][n_vocab])
        #binary_list[dataX[seq_length][n_vocab]] = 1
        dataX[seq_length][n_vocab] = binary_list
        

print(binary_list[0][0])
'''
model = create_model(seq_length, n_vocab)

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

#print(X[0])

model.fit(X, y, epochs = 10, batch_size = 1024, callbacks = [LambdaCallback(on_epoch_end = on_epoch_end), checkpoint])


gene_text()
print("\nDone. ")

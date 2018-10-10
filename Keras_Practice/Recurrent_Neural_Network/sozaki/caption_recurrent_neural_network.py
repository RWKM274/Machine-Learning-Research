import youtube_dl
import re
import os
import webvtt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model

max_len_of_sent = 40

epochs = 50

# variables for testing the model every so often
epochs_until_test = 5
length_of_text_to_generate = 100



# credit: pdemange. From his collector.py
class CaptionCollector:

    def __init__(self, lang='en'):
        self.language = lang
        self.ydlOpts = {'writesubtitles': lang,
                        'skip_download': True,
                        'outtmpl':'subtitles.vtt'}
        self.urlFinder = re.compile('(https?://\S+)')

    def downloadSubs(self, video, filename='subtitles'):
        self.ydlOpts['outtmpl'] = filename+'.vtt'
        with youtube_dl.YoutubeDL(self.ydlOpts) as ydl:
            ydl.download([video])

    def readAllCaptions(self, file):
        captionsList = []
        for caption in webvtt.read(file):
            captionsList.append(caption.text)
        return captionsList

    def formatCaptions(self, captions, replacementDict=None):
        if replacementDict is not None:
            newCaptions = []
            if isinstance(captions, list):
                for caption in captions:
                    if isinstance(replacementDict, dict):
                        for substring, replacement in replacementDict.items():
                            print('Replacing %s with %s' %(substring,replacement))
                            caption = caption.replace(substring, replacement)
                        newCaptions.append(caption)
                    else:
                        print('Replacement dictionary is not in the right format!')
                        break
            return newCaptions
        else:
            print('Nothing to format!')

    def downloadFromList(self, file, subtitleFileName='subtitles'):
        urls = None
        with open(file, 'r') as f:
            urls = self.urlFinder.findall(f.read())
            f.close()

        for i, url in enumerate(urls):
            temp = subtitleFileName+'_'+str(i)
            self.downloadSubs(url, temp)

class directory_of_letters:

    def __init__(self, list):
        self.int_to_char = dict()
        self.char_to_int = dict()
        for i in range(len(list)):
            self.int_to_char[i] = list[i]
            self.char_to_int[list[i]] = i

def prepare_data(list, num_unique_letters, dic_of_letters, steps=3):
    x_raw = []
    y_raw = []
    for i in range(0, len(list) - max_len_of_sent, steps):
        x_raw.append(list[i:i + max_len_of_sent])
        y_raw.append(list[i + max_len_of_sent])
    x_train = np.zeros((len(y_raw), max_len_of_sent, num_unique_letters), dtype=np.bool)
    y_train = np.zeros((len(y_raw), num_unique_letters), dtype=np.bool)

    # converting letters into array of 0 and 1 for the appropriate letter
    for t, sentence in enumerate(x_raw):
        for loc, letters in enumerate(sentence):
            # print(t, loc, letters, dic_of_letters.char_to_int[letters])
            x_train[t][loc][dic_of_letters.char_to_int[letters]] = 1

    for t, sentence in enumerate(y_raw):
        for loc, letters in enumerate(sentence):
            # print(t, loc, letters, dict_of_letters.char_to_int[letters])
            y_train[t][dic_of_letters.char_to_int[letters]] = 1
    return x_train, y_train


def convert_to_text_array(list, steps=3):
    x_raw = []
    for i in range(0, len(list) - max_len_of_sent, steps):
        x_raw.append(list[i:i + max_len_of_sent])
    return x_raw


def convert_to_np_array(text_array):
    x_train = np.zeros((len(text_array), max_len_of_sent, number_of_unique_letters), dtype=np.bool)
    # converting letters into array of 0 and 1 for the appropriate letter
    for t, sentence in enumerate(x_raw):
        for loc, letters in enumerate(sentence):
            # print(t, loc, letters, dic_of_letters.char_to_int[letters])
            x_train[t][loc][dict_of_letters.char_to_int[letters]] = 1


def create_model(num_unique_letters):
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_len_of_sent, num_unique_letters)))
    model.add(Dense(num_unique_letters, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.optimizer.lr = 0.01
    return model


def convert_text_to_array(raw_text, num_unique_letters, dic_of_letters):
    zero_array = np.zeros((len(raw_text), max_len_of_sent, num_unique_letters), dtype=np.bool)

    # converting letters into array of 0 and 1 for the appropriate letter
    for t, sentence in enumerate(raw_text):
        for loc, letters in enumerate(sentence):
            # print(t, loc, letters, dic_of_letters.char_to_int[letters])
            zero_array[t][loc][dic_of_letters.char_to_int[letters]] = 1
    return zero_array


def on_epoch_end(epoch, logs):
    if(epoch % epochs_until_test == 0):
        full_generated_text = ''
        test_text = x_raw[0]
        print('Testing time!')
        print(test_text)
        full_generated_text = full_generated_text + test_text

        for i in range(length_of_text_to_generate):
            print('start: ' + test_text)
            test_array = convert_text_to_array(test_text, number_of_unique_letters, dict_of_letters)
            array_answer = model.predict(test_array, verbose=0)[0]
            added_letter = dict_of_letters.int_to_char[np.argmax(array_answer)]
            # print(added_letter)
            test_text = test_text + added_letter
            test_text = test_text[1:]
            print('ending: ' + test_text)
            full_generated_text = full_generated_text + added_letter
        print(full_generated_text)


# I may need to clean up the data (maybe)
captions_class = CaptionCollector()
all_text = ''
for file_name in os.listdir('.'):
	if(file_name.endswith('en.vtt')):
		caption = captions_class.readAllCaptions(file_name)
		sent = str.join(' ', caption).lower()
		all_text = all_text + sent

test_file = open("all_text.txt", 'w')
test_file.write(all_text)

ordered_list = sorted(list(set(all_text)))
dict_of_letters = directory_of_letters(ordered_list)
number_of_unique_letters = len(set(all_text))
x, y = prepare_data(all_text, number_of_unique_letters, dict_of_letters)

x_raw = convert_to_text_array(all_text)


if __name__ == '__main__':
    captions_class = CaptionCollector()
    # captions_class.downloadFromList("youtube_links_game_theory.txt", "game_theory")
    # captions_class.downloadSubs("https://youtu.be/otwkRq_KnG0", "8-bitryan")
    # caption = captions_class.readAllCaptions('8-bitryan.en.vtt')



    # pass in the sentence and the number of unique letters
    x, y = prepare_data(all_text, number_of_unique_letters, dict_of_letters)
    # model = load_model('caption.h5')
    model = create_model(number_of_unique_letters)
    model.fit(x, y, batch_size=1024, epochs=epochs, verbose=1, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
    model.save('caption.h5')




    # x, y = prepare_data(sent, number_of_unique_letters, dict_of_letters)
    # ordered_list = sorted(list(set(sent)))
    # dict_of_letters = directory_of_letters(ordered_list)
    # number_of_unique_letters = len(set(sent.lower()))




    # time destriputed

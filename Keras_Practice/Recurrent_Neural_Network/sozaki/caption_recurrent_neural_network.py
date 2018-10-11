import youtube_dl
import re
import os
import webvtt
import numpy as np
import random
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import LambdaCallback
from keras.models import Sequential, model_from_json

max_len_of_sent = 40

epochs = 50

# variables for testing the model every so often
epochs_until_test = 5
length_of_text_to_generate = 100

# load model and weights
load_weights_and_model = False

# train model
train_model = False

# build model
build_model = False

# create vtt files using a list_file
vtt_creation_from_list_file = False

# the name of the file with all the youtube links
list_file_name = 'youtube_links_game_theory.txt'

# file name for vtt that contains the youtube captions
multi_vtt_file_name = 'game_theory'

# link to a single youtube video
link_to_youtube = 'https://youtu.be/otwkRq_KnG0'

# single file name for vtt that contains the youtube captions
single_vtt_file_name = '8-bitryan'

# write the combined captions to a file
write_to_txt_file = False

# create vtt for a single youtube caption
single_vtt_creation = False

# specify how many vtt files to create (-1 means create all vtt from the list_file)
list_file_limitor = -1


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


def prepare_data(list, steps=3):
    x_raw = []
    y_raw = []
    for i in range(0, len(list) - max_len_of_sent, steps):
        x_raw.append(list[i:i + max_len_of_sent])
        y_raw.append(list[i + max_len_of_sent])
    x_train = np.zeros((len(y_raw), max_len_of_sent, number_of_unique_letters), dtype=np.bool)
    y_train = np.zeros((len(y_raw), number_of_unique_letters), dtype=np.bool)

    # converting letters into array of 0 and 1 for the appropriate letter
    for t, sentence in enumerate(x_raw):
        for loc, letters in enumerate(sentence):
            x_train[t][loc][char_to_int[letters]] = 1

    for t, sentence in enumerate(y_raw):
        for loc, letters in enumerate(sentence):
            y_train[t][char_to_int[letters]] = 1
    return x_train, y_train


def convert_to_text_array(list, steps=3):
    x_raw = []
    for i in range(0, len(list) - max_len_of_sent, steps):
        x_raw.append(list[i:i + max_len_of_sent])
    return x_raw


def create_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_len_of_sent, number_of_unique_letters)))
    model.add(Dense(number_of_unique_letters, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.optimizer.lr = 0.01
    return model


def convert_text_to_array(raw_text):
    zero_array = np.zeros((1, max_len_of_sent, number_of_unique_letters), dtype=np.bool)

    # converting letters into array of 0 and 1 for the appropriate letter
    for t, sentence in enumerate(raw_text):
            zero_array[0][t][char_to_int[sentence]] = 1
    return zero_array


def on_epoch_end(epoch, logs):
    if(epoch % epochs_until_test == 0):
        full_generated_text = ''
        random_example_index = random.randint(0,len(x_raw))
        test_text = x_raw[random_example_index]
        print('Testing time!')
        full_generated_text = full_generated_text + test_text

        for i in range(length_of_text_to_generate):
            test_array = convert_text_to_array(test_text)
            array_answer = model.predict(test_array, verbose=0)[0]
            added_letter = int_to_char[np.argmax(array_answer)]
            # print(added_letter)
            test_text = test_text + added_letter
            test_text = test_text[1:]
            full_generated_text = full_generated_text + added_letter
        print(full_generated_text)


# always initalize with caption class
captions_class = CaptionCollector()

if vtt_creation_from_list_file:
    # download captions from youtube based on a file of links
    captions_class.downloadFromList(list_file_name, multi_vtt_file_name)

if single_vtt_creation:
    # download caption from a single youtube video
    captions_class.downloadSubs(link_to_youtube, single_vtt_file_name)

if vtt_creation_from_list_file or single_vtt_creation:
    all_text = ''
    for file_name in os.listdir('.'):
        if file_name.endswith('en.vtt'):
            caption = captions_class.readAllCaptions(file_name)
            sent = str.join(' ', caption).lower()
            all_text = all_text + sent

    if write_to_txt_file:
        test_file = open("all_text.txt", 'w')
        test_file.write(all_text)

    # create a dictionary that contains int to char and vice versa
    ordered_list = sorted(list(set(all_text)))
    int_to_char = dict()
    char_to_int = dict()
    for i in range(len(ordered_list)):
        int_to_char[i] = ordered_list[i]
        char_to_int[ordered_list[i]] = i


    number_of_unique_letters = len(set(all_text))
    x, y = prepare_data(all_text)

    # create an array of all prompts (input) to use for testing the model
    x_raw = convert_to_text_array(all_text)

if __name__ == '__main__':

    if build_model:
        model = create_model()

    if train_model:
        model.fit(x, y, batch_size=1024, epochs=epochs, verbose=1, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
        with open('mymodel.json', 'w') as f:
            f.write(model.to_json())
        model.save_weights('caption.h5')

    if load_weights_and_model:
        # load model and weights
        caption_model = open('mymodel.json', 'r')
        model = model_from_json(caption_model.read())
        caption_model.close()
        model.load_weights('caption.h5')

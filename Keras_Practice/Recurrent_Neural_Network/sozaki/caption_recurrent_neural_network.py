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

# load model and weights (note, it doesnt do anything after yet)
load_weights_and_model = False

# build and train the model
build_and_train_model = True

# create vtt files using a list_file
vtt_creation_from_list_file = True

# create vtt for a single youtube caption
single_vtt_creation = False

# the name of the file with all the youtube links
link_list_file_name = 'youtube_links_game_theory.txt'

# file name for vtt that contains the youtube captions
multi_vtt_file_name = 'game_theory'

# link to a single youtube video
link_to_youtube = 'https://youtu.be/otwkRq_KnG0'

# single file name for vtt that contains the youtube captions
single_vtt_file_name = '8-bitryan'

# file name of the file that has all of the captions
file_name_of_all_caption = 'all_text.txt'

# specify how many vtt files to create (-1 means that it will create all vtt from the list_file) with default of 18
list_file_limitor = 22

# steps to move in a text when creating arrays of sentences
steps = 3





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
            if list_file_limitor == -1 or i < list_file_limitor:
                temp = subtitleFileName+'_'+str(i)
                self.downloadSubs(url, temp)

# contains all the captions from the Youtube videos
class CaptionsText:
    def __init__(self):
        self.number_of_unique_letters = 0
        self.x_raw = []
        self.y_raw = []
        self.int_to_char = dict()
        self.char_to_int = dict()
        self.all_caption_string = ''

    def vtt_to_txtfile(self):
        # always initalize with caption class
        captions_class = CaptionCollector()

        if vtt_creation_from_list_file:
            # download captions from youtube based on a file of links
            captions_class.downloadFromList(link_list_file_name, multi_vtt_file_name)

        if single_vtt_creation:
            # download caption from a single youtube video
            captions_class.downloadSubs(link_to_youtube, single_vtt_file_name)

        # move vtt content to a txt file
        all_text = ''
        for file_name in os.listdir('.'):
            if file_name.endswith('en.vtt'):
                print('reading captions from ' + file_name)
                caption = captions_class.readAllCaptions(file_name)
                sent = str.join(' ', caption).lower()
                all_text = all_text + sent

        # save all captions to file
        print('creating ' + file_name_of_all_caption)
        test_file = open(file_name_of_all_caption, 'w')
        test_file.write(all_text)
        test_file.close()

    def create_dictionary_and_prepare_data(self):
        if vtt_creation_from_list_file or single_vtt_creation:
            self.vtt_to_txtfile()

        # opens file with all of the captions
        print('reading from ' + file_name_of_all_caption)
        all_caption_text_file = open(file_name_of_all_caption, 'r')
        self.all_caption_string = all_caption_text_file.read()
        all_caption_text_file.close()

        # checks to make sure that the content was loaded
        if self.all_caption_string != '':


            # create a dictionary that contains int to char and vice versa
            ordered_list = sorted(list(set(self.all_caption_string)))
            for i in range(len(ordered_list)):
                self.int_to_char[i] = ordered_list[i]
                self.char_to_int[ordered_list[i]] = i

            # creating raw text dataset and training dataset
            self.number_of_unique_letters = len(set(self.all_caption_string))
            # print(self.number_of_unique_letters)

            """ Going through the full caption string and creating an array that contains
                raw text that is max_len_of_sent in length.
            """
            for i in range(0, len(self.all_caption_string) - max_len_of_sent, steps):
                self.x_raw.append(self.all_caption_string[i:i + max_len_of_sent])
                self.y_raw.append(self.all_caption_string[i + max_len_of_sent])

            """ creating numpy array with the x training set to have a shape of (length of all the samples,
                length of a sentence, length of all the possible unique words in the text). And the y training set
                has a shape of (length of all the samples, length of all the possible unique words in the text).
            """
            self.x_train = np.zeros((len(self.x_raw), max_len_of_sent, self.number_of_unique_letters), dtype=np.bool)
            self.y_train = np.zeros((len(self.x_raw), self.number_of_unique_letters), dtype=np.bool)

            """ Converting letters into array of 0 and 1 (where 1 is the location that is associated with that letter)
                with a length equal to all the unique characters. It does that for the x and y (input and output
                respectively).
            """
            for t, sentence in enumerate(self.x_raw):
                for loc, letters in enumerate(sentence):
                    self.x_train[t][loc][self.char_to_int[letters]] = 1

            for t, sentence in enumerate(self.y_raw):
                for loc, letters in enumerate(sentence):
                    self.y_train[t][self.char_to_int[letters]] = 1
        else:
            print('Warning! There is no captions. (you may not have any en.vtt files.)')

    def on_epoch_end(self, epoch, logs):
        print('working')
        if epoch % epochs_until_test == 0:
            full_generated_text = ''

            # picks a random sentence from the array of sentences
            random_example_index = random.randint(0, len(self.x_raw))
            test_text = self.x_raw[random_example_index]
            print('Testing time!')
            full_generated_text = full_generated_text + test_text

            # generates length_of_text_to_generate letters and adds that to the original sentence
            for i in range(length_of_text_to_generate):

                # converting raw text to numpy array
                test_array = self.convert_text_to_array(test_text)

                # using numpy array, it will predict what the next letter will be
                array_answer = model.predict(test_array, verbose=0)[0]
                added_letter = self.int_to_char[np.argmax(array_answer)]
                # print(added_letter)

                # add the letter to the sentence and then take everything but the first letter and feed that back in
                test_text = test_text + added_letter
                test_text = test_text[1:]

                # add the new letter to the full text
                full_generated_text = full_generated_text + added_letter

            print(full_generated_text)

    def convert_text_to_array(self, raw_text):
        zero_array = np.zeros((1, max_len_of_sent, self.number_of_unique_letters), dtype=np.bool)

        # converting letters into array of 0 and 1 for the appropriate letter
        for t, sentence in enumerate(raw_text):
            zero_array[0][t][self.char_to_int[sentence]] = 1
        return zero_array





def create_model(number_of_unique_letters):
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_len_of_sent, number_of_unique_letters)))
    model.add(Dense(number_of_unique_letters, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.optimizer.lr = 0.01
    return model


if __name__ == '__main__':

    # preparing the training set
    caption_text = CaptionsText()
    caption_text.create_dictionary_and_prepare_data()

    # checks to make sure that the content was loaded
    if build_and_train_model and caption_text.all_caption_string != '':
        model = create_model(caption_text.number_of_unique_letters)
        model.fit(caption_text.x_train, caption_text.y_train, batch_size=1024, epochs=epochs, verbose=1, callbacks=[LambdaCallback(
            on_epoch_end=caption_text.on_epoch_end)])
        with open('mymodel.json', 'w') as f:
            f.write(model.to_json())
        model.save_weights('caption.h5')

    # checks to make sure that the content was loaded
    if load_weights_and_model:
        # load model and weights
        caption_model = open('mymodel.json', 'r')
        model = model_from_json(caption_model.read())
        caption_model.close()
        model.load_weights('caption.h5')

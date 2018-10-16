
from keras.models import model_from_json
from flask import request
from flask import jsonify
from flask import Flask, render_template
import tensorflow as tf
import random
import numpy as np



app = Flask(__name__)

class text_generator:
    def __init__(self):
        caption_model = open('mymodel.json', 'r')
        self.model = model_from_json(caption_model.read())
        caption_model.close()
        self.model.load_weights('caption.h5')
        print("Loaded model!")

        # file name of the file that has all of the captions
        self.file_name_of_all_caption = 'all_text.txt'


        # initialize all values
        self.int_to_char = dict()
        self.char_to_int = dict()
        self.x_raw = []
        # steps to move in a text when creating arrays of sentences
        self.steps = 3
        self.length_of_text_to_generate = 200

        # will be filled later
        self.number_of_unique_letters = 0

        # do not change this
        self.max_len_of_sent = 40

        # open the file with the caption. Then create dictionary and list with seeds of 'sentences'
        self.creating_array_of_sentences()

        # determines how risky the network will be in its predictions
        self.temperature = .2


    def creating_array_of_sentences(self):
        all_caption_text_file = open(self.file_name_of_all_caption, 'r')
        all_caption_string = all_caption_text_file.read()

        # create a dictionary that contains int to char and vice versa
        ordered_list = sorted(list(set(all_caption_string)))
        for i in range(len(ordered_list)):
            self.int_to_char[i] = ordered_list[i]
            self.char_to_int[ordered_list[i]] = i


        # creating raw text dataset and training dataset
        self.number_of_unique_letters = len(set(all_caption_string))
        print(self.number_of_unique_letters)

        """ Going through the full caption string and creating an array that contains
            raw text that is max_len_of_sent in length.
        """
        for i in range(0, len(all_caption_string) - self.max_len_of_sent, self.steps):
            self.x_raw.append(all_caption_string[i:i + self.max_len_of_sent])

    def generating_text(self):
        # generating text


        full_generated_text = ''
        # picks a random sentence from the array of sentences
        random_example_index = random.randint(0, len(self.x_raw))
        test_text = self.x_raw[random_example_index]
        print('Time to generate text!')
        full_generated_text = full_generated_text + test_text


        # generates length_of_text_to_generate letters and adds that to the original sentence
        for i in range(self.length_of_text_to_generate):
            # converting raw text to numpy array
            test_array = self.convert_text_to_array(test_text)

            # using numpy array, it will predict what the next letter will be (using sample to me more daring
            array_answer = self.model.predict(test_array, verbose=0)[0]
            added_letter = self.int_to_char[np.argmax(array_answer)]

            # add the letter to the sentence and then take everything but the first letter and feed that back in
            test_text = test_text + added_letter
            test_text = test_text[1:]

            # add the new letter to the full text
            full_generated_text = full_generated_text + added_letter

        print(full_generated_text)
        return full_generated_text

    def convert_text_to_array(self, raw_text):
        zero_array = np.zeros((1, self.max_len_of_sent, self.number_of_unique_letters), dtype=np.bool)

        # converting letters into array of 0 and 1 for the appropriate letter
        for t, sentence in enumerate(raw_text):
            zero_array[0][t][self.char_to_int[sentence]] = 1
        return zero_array

def get_model():
    global model
    global graph
    graph = tf.get_default_graph()

print('loading model and all associated lists and dictionaries')
get_model()
text_gen = text_generator()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/generate_text", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    with graph.as_default():
        generated_txt = text_gen.generating_text()

        # clips the beginning word because it might have been cut off
        position_of_space = generated_txt.find(' ')
        print('position where the space starts: ' + str(position_of_space))
        if position_of_space != 0:
            generated_txt = generated_txt[position_of_space:]

    response = {
        'prediction' : generated_txt
    }

    return jsonify(response)

app.run()
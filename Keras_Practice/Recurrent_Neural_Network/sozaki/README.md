# Rucurrent Neural Network

## Overview
It will grab the captions from Youtube and create a txt file that contains all of the captions. Then it will train using those captions and every so often it will get a seed (sentence) and then continues that sentence with words that it thinks will work there.

## Setup and Run this RNN
If you have not created your own vtt files from Youtube:
* Go into `caption_recurrent_neural_network.py`. At the top of the file, change all of the boolean values to False except for `vtt_creation_from_list_file`. Also, change the string of `link_list_file_name` to match the name of the file that contains all the links to Youtube videos you want to take.
After you have the vtt files:
* Simply change the `vtt_creation_from_list_file` back to False and then change `build_and_train_model` to True then in the terminal run `python3 caption_recurrent_neural_network.py`
After it is done training it will save its weights in `caption.h5` and saves the rest into `mymodel.json`



## Note
The Practice_RNN contains the neural network that I created to learn RNN.

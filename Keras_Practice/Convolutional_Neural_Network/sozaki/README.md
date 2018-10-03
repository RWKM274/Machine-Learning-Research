# Reconizing types of Fruit

## Dataset
Retrieved dataset from [Kaggle](https://www.kaggle.com/moltean/fruits)

## Implementation
I used Keras' generators to grab the appropirate pictures of fruits in their respective directories and then classify by the names of the directories.

## Information
I have trained the neural network using less data to keep the time to train the CNN down.

## Running this script
- You need to have the Keras package for Python3 already installed before starting this program.
- Go into the script and at the top there will be booleans, which you will need to change depending on whether you want to create your create a new neural network or use the default network.
- **Currently**, the setting is going to run the current neural network if it exists.

## Find pictures from Google
Run `pip3 install google_images_download` if you have pip3.
Then run `googleimagesdownload -k "name of the images you want to seach" -l 20` (20 is the limit of images it downloads)
For example,
`googleimagesdownload -k "a bunch of apples" -l 20` searches for 'a bunch of apples' and only downloads 20 images

**for more information** please visit creator's [git repo](https://github.com/hardikvasa/google-images-download)

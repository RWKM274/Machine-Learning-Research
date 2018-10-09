# Convolutional Neural Network (CNN)
Type of Neural Network specifically used for recongnizing images.

## Concept
Takes in an image and of a chosen scale. It uses a filter (yellow) and uses an acumulation of the filter and that will be a part of the next window called a kernal window.<br/>
![Example of a filter](https://i.stack.imgur.com/9Iu89.gif)<br/>
This allows the Neural Network to generals the pixels to allow for corner cases.<br/>
Max Pooling helps to condense a picture to keep the neurons count down. It is similar to the concept above. A filter is used to find the largest number and puts that into the kernal window. Then the stride determines the pixels or amount it will move along the image or matrix

![Example of Max Pooling](https://adeshpande3.github.io/assets/Cover2nd.png)<br/>


#### [For More Information](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/)

## Issues
If you get this error that looks something like this: `Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA keras` you need to do this to get rid of the warning `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'`.<br/>
[For more information visite this stackoverflow page](https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u)

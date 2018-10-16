from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array,ImageDataGenerator
from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
from PIL import Image 
import base64
import io 
import numpy as np
import re
import tensorflow as tf
from build_model import buildModel
#K.set_learning_phase(1)
app = Flask(__name__)

model = buildModel()

def loadWeights(weightsFile):
	model.load_weights(weightsFile)
	return model

def preprocessImage(image, size):
	if image.mode != 'RGB':
		image = image.convert('RGB')
	image = image.resize(size)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	return image

model = loadModel('weights.h5')
print('Model has been loaded!')
global graph
graph = tf.get_default_graph() 
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/sample', methods=['POST'])
def sample():
	message = request.get_json(force=True)
	encoded = re.sub('^data:image/.+;base64,', '', message['image'])
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed = preprocessImage(image, (224,224))
	print('Predicting!')
	with graph.as_default():
		answer = model.predict(processed)
	print(answer)
	response = {
		'prediction':{
			'cat': float(answer[0][0]),
			'dog': float(answer[0][1])
		}
	}
	return jsonify(response)

app.run(host='127.0.0.1')

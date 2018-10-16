![Flask Logo](http://flask.pocoo.org/static/logo/flask.png)

# Flask

A micro framework used to develop Python webservers. With this, neural networks can be served on a webserver that we make with flask since both of them run on Python.

## Installing Flask in Python

```bash
>$ pip install flask
```
## Getting Started with Flask

To get started with flask, it first needs to be imported like this: 

```Python 
from flask import Flask

#Start a flask instance

app = Flask(__name__)
```

Then, to define it's webpages, we specify the routes to the pages and the functions that handle the pages directly after.

```Python
#When the base website is visited (ex. www.example.com), the function index() will run and return the html <h1>hello world!</h1>

@app.route('/')
def index():
  return '<h1>hello world!</h1>'
```

Finally, we run the Flask server like so: 

```Python 
app.run(host='127.0.0.1', port=5000)
```

And in the terminal:
```bash
>$ python thepythonprogram.py
```

## Handling Different Methods

If we want a specific webpage to have the ability to handle POST requests, we can specify that in Flask:

```Python
@app.route('/example', methods=['POST','GET'])
def postExample(): 
  ....
```
## Rendering External HTML Web Pages

To do this, a folder called "templates" must be created for Flask within the same directory as the Flask program.

```bash
>$ mkdir templates
```

Then, within the folder you can put static web pages to be rendered, or accessed, by the Flask program.

In order to render these web pages within a Flask program, it's done like this: 

```Python
from flask import render_template
...
@app.route('/example', methods=['POST', 'GET'])
def exampleFunc():
  return render_template('example.html')
```
## Handling Requests in Flask

To handle post requests, we can do so like this: 

```Python
from flask import request, jsonify
...
@app.route('/example', methods=['POST'])
def exampleFunc():
	message = request.get_json(force=True)
	name = message['name']
	response = {
		'greeting': 'Hello, ' + name + '!'
	}
  return jsonify(response)
```

## Links
Here is the link to their [website](http://flask.pocoo.org/) and [github](https://github.com/pallets/flask).

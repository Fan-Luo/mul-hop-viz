from flask import Flask, request
from flask import render_template
from flask.json import jsonify
from flask_cors import CORS
import string, random
from sklearn.manifold import TSNE
import json
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route("/simple")
def simple():
  return render_template("simple_d3.html")

@app.route("/fetch_first_example", methods=['GET'])
def fetch_first_example():
  response = jsonify("first")
  return response

@app.route("/fetch_next_example", methods=['POST'])
def fetch_next_example():
  words = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
  
  input_data = request.json
  print(input_data)

  word_to_return = words[int(input_data)]

  input_data = input_data + 1

  if input_data == 10 :
    input_data = 0
  #

  response = jsonify(
    {"word" : word_to_return,
     "counter" : int(input_data) })
  return response





if (__name__ == '__main__') :
  app.run(debug = True)
#

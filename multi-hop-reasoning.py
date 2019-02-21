from flask import Flask, request
from flask import render_template
from flask.json import jsonify
import string, random
from sklearn.manifold import TSNE
import json
import numpy as np

app = Flask(__name__)

@app.route("/")
def user():
  user_dict = {'first': 'Fan', 'last': 'Luo'}
  return render_template("user.html", user_dict = user_dict.values())
  
@app.route("/tsne")
def tsne():
    with open('static/knowledge_fact_fan_49.json') as json_file:  
        data = json.load(json_file)
        X_embedded = TSNE(n_components=2, perplexity=40, n_iter=300).fit_transform(data['mat'])

    return render_template("user.html", X_embedded = X_embedded.tolist())
    #return render_template("user.html", X_embedded = np.array2string(X_embedded, separator=', '))

 

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
  # title = 'Visualization for Multi-hop Reasoning'
  user_dict = {'first': 'Robert', 'last': 'Chang', 'twitter_handle': '@_rchang'}
  return render_template("user.html", user_dict = user_dict.values())
  
# @app.route("/data")

# def data():
#     num_facts = request.args.get('text')
#     text = loadfacts(int(num_facts))

#     # print(int(num_facts))
#     # print(text)

#     return render_template("user.html", text = text)


# def read_file(datafile):
#     with open(datafile, 'r') as f:
#         lines = [line for line in f]
#         return lines


# def loadfacts(n):
#     facts = read_file('openbook.txt')

#     f = []
#     for i in range(int(n)):
#         f_id = random.randint(0, len(facts))
#         f.append(facts[f_id])
#     return f



@app.route("/tsne")
def tsne():
    with open('static/knowledge_fact_fan_49.json') as json_file:  
        data = json.load(json_file)
        X_embedded = TSNE(n_components=2, perplexity=40, n_iter=300).fit_transform(data['mat'])

    return render_template("user.html", X_embedded = X_embedded.tolist())
    #return render_template("user.html", X_embedded = np.array2string(X_embedded, separator=', '))

 
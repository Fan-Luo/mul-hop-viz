from flask import Flask, request
from flask import render_template
from flask.json import jsonify
# from flask_cors import CORS
# from serve import get_model_api
import string, random
from sklearn.manifold import TSNE
import json
import numpy as np
from DyMemNet_GRUReasoner import *

app = Flask(__name__)
# CORS(app) # needed for cross-domain requests, allow everything by default
# model_api = get_model_api()



@app.route("/")
def user():
  user_dict = {'first': 'Fan', 'last': 'Luo'}
  return render_template("user.html", user_dict = user_dict.values())
  
# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

@app.route("/tsne")
def tsne():
    with open('static/knowledge_fact_fan_49.json') as json_file:  
        data = json.load(json_file)
        X_embedded = TSNE(n_components=2, perplexity=40, n_iter=300).fit_transform(data['mat'])

    return render_template("user.html", X_embedded = X_embedded.tolist())
    #return render_template("user.html", X_embedded = np.array2string(X_embedded, separator=', '))

@app.route("/annotation")
def annotation():

    return render_template("annotation.html")


@app.route("/fetch_first_example", methods=['GET'])
def fetch_first_example():

    
    instance = trainer.instances_train[0]
    choice_id = np.argmax(np.array(instance.target))
    # output, att_scores, question_text, candidate_text, facts_text = classifier.train_interactive_forward(instance, 0)
    # output is the confidence of the correct answer

    print('=================================')
    print('DEBUG question text:',instance.question_text)
    print('DEBUG choice text:', instance.choices_text[choice_id])

    output, scores, question_text, choice_text, facts_text= trainer.train_interactive_forward(instance, choice_id)

    web_response = {}
    web_response['output_score'] = output
    web_response['fact_scores'] = scores
    web_response['question_text'] = question_text
    web_response['choice_text'] = choice_text
    web_response['facts_text'] = facts_text

    response = jsonify(web_response)     
    return response


@app.route("/annotation_and_next_example", methods=['POST'])
def annotation_and_next_example():
    #-----------train the current instance----------------
    #read from the POST what the selected entries were
    user_annotations = request.json               # input is the selected facts with score 1
    loss, predict = trainer.train_interactive_backward(0, user_annotations)   # back propagate error according to user annotated facts
    # caution: the current model only trained on the selected fact but not the actual choice, because currently we only show the question and the correct choice.
    
    #-----------reset the instance counter to handle the next instance---------------
    trainer.training_instance_counter+=1
    instance_index = trainer.training_instance_counter

    instance = trainer.instances_train[instance_index]
    choice_id = np.argmax(np.array(instance.target))

    print('=================================')
    print('DEBUG question text:',instance.question_text)
    print('DEBUG choice text:', instance.choices_text[choice_id])

    # output format is the same as the get first example
    output, scores, question_text, choice_text, facts_text= trainer.train_interactive_forward(instance, choice_id)

    web_response = {}
    web_response['output_score'] = output
    web_response['fact_scores'] = scores
    web_response['question_text'] = question_text
    web_response['choice_text'] = choice_text
    web_response['facts_text'] = facts_text

    response = jsonify(web_response)  

    return response

# @app.route("/selection", methods=['POST'])
# def selection():
#     return render_template("selection.html" )


 
if (__name__ == '__main__'):
    global trainer
    trainer = DyMemNet_Trainer()
    app.run(debug = True)
	
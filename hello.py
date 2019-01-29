from flask import Flask, request
from flask import render_template
import string, random

app = Flask(__name__)

@app.route("/")
def user():
  # title = 'Visualization for Multi-hop Reasoning'
  user_dict = {'first': 'Robert', 'last': 'Chang', 'twitter_handle': '@_rchang'}
  return render_template("user.html", user_dict = user_dict.values())
	# return render_template("user.html")

@app.route("/data")

def data():
	num_facts = request.args.get('text')
	text = loadfacts(int(num_facts))

	# print(int(num_facts))
	# print(text)

	return render_template("user.html", text = text)


def read_file(datafile):
    with open(datafile, 'r') as f:
	    lines = [line for line in f]
	    return lines


def loadfacts(n):
	facts = read_file('openbook.txt')

	f = []
	for i in range(int(n)):
		f_id = random.randint(0, len(facts))
		f.append(facts[f_id])
	return f

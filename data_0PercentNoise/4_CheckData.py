
import numpy as np

import pickle

class Instance:
    def __init__(self, target,  question_text, choices_text, fact_text, knowledge_text):
        self.question_text = question_text
        self.choices_text = choices_text
        self.science_fact_text = fact_text
        self.knowledge_fact_text = knowledge_text
        self.target = target
        self.question_id = 0

with open('MemNet_DevQuestions_Pos.pickle', 'rb') as input_file:
    instances_train = pickle.load(input_file)

for instance in instances_train:
    print('--------new instance---------')
    print(instance.question_text)
    print(instance.choices_text)
    print(instance.science_fact_text)
    print(instance.knowledge_fact_text)
    print(len(instance.knowledge_fact_text))
    input('press enter to continue')

import pickle

class Instance:
    def __init__(self, target,  question_text, choices_text, fact_text, knowledge_text):
        self.question_text = question_text
        self.choices_text = choices_text
        self.science_fact_text = fact_text
        self.knowledge_fact_text = knowledge_text
        self.target = target
        self.question_id = 0

class Annotation():
    def __init__(self, question_id, annotations, new_facts):
        self.question_id = question_id
        self.annotations = annotations
        self.new_facts = new_facts

with open('zhengzhongliang_2019-04-09_1618.pickle', 'rb') as f:
    pickle_file = pickle.load(f)
    print(pickle_file)
    pickle_file = pickle.load(f)
    print(pickle_file)
    pickle_file = pickle.load(f)
    print(pickle_file)
    


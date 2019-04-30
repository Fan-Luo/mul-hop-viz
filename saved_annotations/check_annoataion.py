import pickle

class Instance:
    def __init__(self, target,  question_text, choices_text, fact_text, knowledge_text):
        self.question_text = question_text
        self.choices_text = choices_text
        self.science_fact_text = fact_text
        self.knowledge_fact_text = knowledge_text
        self.target = target
        self.question_id = 0

data = []
with open('fan_2019-04-10_1729.pickle', 'rb') as fr:
    try:
        while True:
            data.append(pickle.load(fr))
    except EOFError:
        pass

print(len(data))
for instance in data:
	print(instance["question_id"],instance["annotation"],instance["new_facts"])

train_path = '../data_0PercentNoise/MemNet_TrainQuestions_Pos.pickle'
with open(train_path, 'rb') as input_file:
    instances_train = pickle.load(input_file)
    instances_train =  instances_train[partition*50:(partition+1)*50]
    for ins in instances_train:
        print(ins.question_text)
        print(ins.choices_text)
        print(ins.science_fact_text)
        print(ins.knowledge_fact_text)
        
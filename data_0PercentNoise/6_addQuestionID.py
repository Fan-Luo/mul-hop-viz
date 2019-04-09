import pickle

class Instance:
    def __init__(self, target,  question_text, choices_text, fact_text, knowledge_text):
        self.question_text = question_text
        self.choices_text = choices_text
        self.science_fact_text = fact_text
        self.knowledge_fact_text = knowledge_text
        self.target = target
        self.question_id = 0

train_path = 'MemNet_TrainQuestions_Pos.pickle'
dev_path = 'MemNet_DevQuestions_Pos.pickle'
test_path = 'MemNet_TestQuestions_Pos.pickle'



with open(train_path, 'rb') as input_file:
    instances_train = pickle.load(input_file)
    
with open(dev_path, 'rb') as input_file:
    instances_dev = pickle.load(input_file)
    
with open(test_path, 'rb') as input_file:
    instances_test = pickle.load(input_file)

for i, instance in enumerate(instances_train):
    instance.question_id = i
    
for i, instance in enumerate(instances_dev):
    instance.question_id = i
    
for i, instance in enumerate(instances_test):
    instance.question_id = i
    
    
with open(train_path, 'wb') as handle:
    pickle.dump(instances_train, handle)
    
with open(dev_path, 'wb') as handle:
    pickle.dump(instances_dev, handle)
    
with open(test_path, 'wb') as handle:
    pickle.dump(instances_dev, handle)
    




from DyMemNet_GRUReasoner import *

class Instance:
    def __init__(self, target,  question_text, choices_text, fact_text, knowledge_text):
        self.question_text = question_text
        self.choices_text = choices_text
        self.science_fact_text = fact_text
        self.knowledge_fact_text = knowledge_text
        self.target = target
        self.question_id = 0
        self.model_confidence = 0

classifier = torch.load('saved_model/MemNet_Trained_Epoch_0')
classifier.n_hop = 4

train_path = 'data_0PercentNoise/MemNet_TrainQuestions_Pos.pickle'
dev_path = 'data_0PercentNoise/MemNet_DevQuestions_Pos.pickle'
test_path = 'data_0PercentNoise/MemNet_TestQuestions_Pos.pickle'

with open(train_path, 'rb') as input_file:
    instances_train = pickle.load(input_file)

with open(dev_path, 'rb') as input_file:
    instances_dev = pickle.load(input_file)

with open(test_path, 'rb') as input_file:
    instances_test = pickle.load(input_file)


print("==========train===========")
epoch_loss = 0
correct_count = 0
dev_loss = 0
for i, instance in enumerate(instances_train):

    outputs = {}
    for choice_id in np.arange(4):
        #if data_noise_percent=='0':
        outputs[choice_id],_ = classifier(instance.question_text, instance.choices_text[choice_id], instance.science_fact_text, instance.knowledge_fact_text, [[],[]])
        #elif data_noise_percent=='30':
        #    outputs[choice_id] = self.classifier(instance.question_text, instance.choices_text[choice_id], instance.science_fact_text[choice_id], instance.knowledge_fact_text[choice_id])

    output = torch.cat([outputs[0], outputs[1], outputs[2], outputs[3]], dim=0).view(1, 4)
    answers = torch.argmax(torch.tensor(instance.target))

    instance.model_confidence= output[0,answers].detach().item()

    loss, pred = classifier.get_loss(output, answers)
    if pred == answers:
        correct_count += 1

    dev_loss += loss
    # input('press enter to continue')
    if (i - 1) % 200 == 0 and i!=1:
        print('sample ', i, 'average training loss:', dev_loss / 200)
        dev_loss = 0
print('dev acc:', correct_count / 1.0 / len(instances_train))


with open(train_path, "wb") as output_file:
    pickle.dump(instances_train, output_file)


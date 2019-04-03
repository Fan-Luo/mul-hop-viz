#from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import pickle
import time
from random import shuffle
import sys
import os

print("pytorch version:", torch.__version__)

print('threads before set:',torch.get_num_threads())
torch.set_num_threads(1)
print('threads after set:',torch.get_num_threads())

# arch_name = sys.argv[0][:-3]
# data_noise_percent = sys.argv[1]
# n_sci_fact = sys.argv[2]
# n_know_fact = sys.argv[3]

# print('arch name:', arch_name)
# print('data_noise_percent:', data_noise_percent,  '\tn_sci_fact:', n_sci_fact, '\tn_know_fact:',n_know_fact)

#dict_path = "/work/zhengzhongliang/DyMemNet_2019/GloveDict/glove.840B.300d.pickle"

dict_path = "/Users/zhengzhongliang/PycharmProjects/DyMemNet/Glove_Embedding/glove.840B.300d.pickle"
#dict_path = "glove.840B.300d.pickle"

with open(dict_path, 'rb') as input_file:
    glove_dict = pickle.load(input_file)


class InputModule(nn.Module):
    def __init__(self, hidden_size=100):
        super(InputModule,self).__init__()
        self.gru = nn.GRU(input_size=300, hidden_size=hidden_size, num_layers=1, bidirectional = True, dropout=0.0)

        # the hidden_size is determined when the InputModule is instantiated.

    def forward(self, input_text):
        # input text should be a nested list, n_sentences * batch_size=1 * sentence_list. e.g., [['first','sentence','.']]
        input_vecs = list([])
        for word in input_text:
            if word in glove_dict:
                input_vecs.append(torch.tensor(glove_dict[word],dtype = torch.float32))
            else:
                input_vecs.append(torch.tensor(np.ones(300)*0.1,dtype = torch.float32))


        seq_len = len(input_vecs)


        input_tensor = torch.stack(input_vecs)
        seq_len, embd_size = input_tensor.size()
        input_tensor = input_tensor.view(seq_len,1,embd_size)

        gru_out, _ = self.gru(input_tensor)

        seq_len, batch_size, dim_embd = gru_out.size()
        gru_out = gru_out.view(batch_size, seq_len, dim_embd)

        r_ctx, _ = torch.max(gru_out,dim=1)
        #print(maxVal.size())
        # maxpool = torch.nn.FractionalMaxPool2d((seq_len,1), output_size = (batch_size, 1, dim_embd))
        # r_ctx = maxpool(gru_out)

        # print(r_ctx.size())
        # print("r_ctx first element:", r_ctx[0,0,0])
        # maxVal, _ = torch.max(gru_out,dim=0)
        # print("gru out max value:", maxVal)
        #input('press enter to continue')


        return torch.squeeze(r_ctx)


class Classifier(nn.Module):
    def __init__(self, hidden_size=200, gru_hidden_size=200, output_size = 1, n_hop = 4):  #default hops:3
        super(Classifier, self).__init__()
        self.inputModule = InputModule(hidden_size=hidden_size)

        self.linear_extract_1 = nn.Linear(hidden_size*12, hidden_size)
        self.linear_extract_2 = nn.Linear(hidden_size,1)
        self.linear_answer = nn.Linear(gru_hidden_size, 1)
        self.memory_mapping = nn.Linear(gru_hidden_size, hidden_size*2)

        # define GRU weights:
        self.W_z = nn.Linear(hidden_size*14, gru_hidden_size)
        self.U_z = nn.Linear(gru_hidden_size, gru_hidden_size)

        self.W_r = nn.Linear(hidden_size * 14, gru_hidden_size)
        self.U_r = nn.Linear(gru_hidden_size, gru_hidden_size)

        self.W_h = nn.Linear(hidden_size * 14, gru_hidden_size)
        self.U_h = nn.Linear(gru_hidden_size, gru_hidden_size)

        self.gru_hidden_size =gru_hidden_size
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.n_hop = n_hop


    def forward(self, question_text, candidate_text, sci_text, knowledge_text):
        # get encoded question, choice and knowledge base
        question_vec = self.inputModule.forward(question_text)
        candidate_vec = \
            self.inputModule.forward(candidate_text)

        hidden_size = candidate_vec.size()
        #if data_noise_percent=='0':
        knowledge_fact = list([])
        for single_fact in knowledge_text[-10:]:
            knowledge_fact.append(self.inputModule.forward(single_fact))
        knowledge_fact.append(self.inputModule.forward(sci_text))
        # elif data_noise_percent=='30':
        #     knowledge_fact = list([])
        #     if len(knowledge_text)>0:
        #         for single_fact in knowledge_text[-min(len(knowledge_text), int(n_know_fact)):]:
        #             knowledge_fact.append(self.inputModule.forward(single_fact))
        #     else:
        #         knowledge_fact.append(torch.tensor(np.zeros(hidden_size), dtype = torch.float32))
        #     sci_fact = list([])
        #     if len(sci_text)>0:
        #         for single_fact in sci_text[-min(len(sci_text), int(n_sci_fact)):]:
        #             sci_fact.append(self.inputModule.forward(single_fact))
        #     else:
        #         sci_fact.append(torch.tensor(np.zeros(hidden_size), dtype = torch.float32))
        #     knowledge_fact.extend(sci_fact)

        knowledge_vecs = torch.stack(knowledge_fact)

        # print('knowledge vec shape:', knowledge_vecs.size())
        # input('press enter to continue')

        n_fact, ctx_dim = knowledge_vecs.size()

        # multi-hop reasoning

        # memory_vec size: 200
        memory_vec = torch.tensor(np.zeros(self.gru_hidden_size),dtype = torch.float32)
        att_scores = list([])
        for hop in np.arange(self.n_hop):

            # ref_vec size: 6*1*1200
            mem_interface = self.memory_mapping(memory_vec)
            ref_vec = torch.cat([question_vec.expand_as(knowledge_vecs)*knowledge_vecs,
                                 candidate_vec.expand_as(knowledge_vecs)*knowledge_vecs,
                                 mem_interface.expand_as(knowledge_vecs)*knowledge_vecs,
                                 torch.abs(question_vec.expand_as(knowledge_vecs)-knowledge_vecs),
                                 torch.abs(candidate_vec.expand_as(knowledge_vecs)-knowledge_vecs),
                                 torch.abs(mem_interface.expand_as(knowledge_vecs)-knowledge_vecs)],dim=1)
            # retrieved_vec size: 200
            att_score_extract = F.softmax(self.linear_extract_2(F.relu(self.linear_extract_1(ref_vec))), dim=0)
            retrieved_tensor = att_score_extract*knowledge_vecs
            retrieved_vec = torch.sum(retrieved_tensor, dim=0)

            att_scores.append(att_score_extract.squeeze())

            mem_ref_input = torch.cat([question_vec,
                                          candidate_vec,
                                          retrieved_vec,
                                          question_vec*retrieved_vec,
                                          candidate_vec*retrieved_vec,
                                          torch.abs(question_vec-retrieved_vec),
                                          torch.abs(candidate_vec-retrieved_vec)], dim=0)
            z_t = F.sigmoid(self.W_z(mem_ref_input)+self.U_z(memory_vec))
            r_t = F.sigmoid(self.W_r(mem_ref_input)+self.U_r(memory_vec))
            memory_vec = (1-z_t)*memory_vec +z_t*F.tanh(self.W_h(mem_ref_input)+self.U_h(r_t*memory_vec))

            # CHECK THE ELEMENTS OF z_t and r_t   !!!!!!!

        final_score = self.linear_answer(memory_vec)

        return final_score, att_scores

    def get_loss(self, output, targets):
        loss = self.criterion(output, targets.view(1))
        preds = F.softmax(output, dim=1)
        _, pred_ids = torch.max(preds, dim=1)
        return loss, pred_ids

    def get_loss_interactive(self, outputs, targets, att_score, annotated_sen):
        #prediction_loss = self.criterion(output, targets.view(1))


        #print("What the hell is the output!", outputs)
        #print("What the hell is the output!", outputs.size())

        selection_loss =0 
        selection_loss+=self.criterion(outputs.view(1,4), torch.tensor(targets).view(1))
        for hop in np.arange(len(annotated_sen)):
            adjusted_annotation = 10-annotated_sen[hop]
            selection_loss+=self.criterion(att_score[hop].view(1,11), torch.tensor(adjusted_annotation).view(1))
        _, preds=torch.max(outputs.view(1,4), dim=1)
        #preds = F.softmax(output, dim=1)
        #_, pred_ids = torch.max(preds, dim=1)
        #return prediction_loss+selection_loss, pred_ids
        return selection_loss, preds



class Instance:
    def __init__(self, target,  question_text, choices_text, fact_text, knowledge_text):
        self.question_text = question_text
        self.choices_text = choices_text
        self.science_fact_text = fact_text
        self.knowledge_fact_text = knowledge_text
        self.target = target
        self.human_annotations = 0


class DyMemNet_Trainer():
    def __init__(self, load_model=True, train_path = 'data_0PercentNoise/MemNet_TrainQuestions_Pos.pickle', dev_path = 'data_0PercentNoise/MemNet_DevQuestions_Pos.pickle'):
        if load_model:
            self.classifier = torch.load('saved_model/MemNet_Trained_Epoch_0')
        else:
            self.classifier = Classifier()
        self.optim =  torch.optim.Adam(self.classifier.parameters())

        with open(train_path, 'rb') as input_file:
            self.instances_train = pickle.load(input_file)

        shuffle(self.instances_train)

        with open(dev_path, 'rb') as input_file:
            self.instances_dev = pickle.load(input_file)

        self.att_scores = 0
        self.outputs = {}

        self.training_instance_counter = 0 



        # with open('data_'+data_noise_percent+'PercentNoise/MemNet_TestQuestions_Pos.pickle', 'rb') as input_file:
        #     self.instances_test = pickle.load(input_file)

    def trainEpoch(self, n_epoch=1, n_samples = 400):
        save_folder_path = 'DyMemNet_GRUReasoner'
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        

        model_results = np.zeros((n_epoch, 4))
        for epoch in np.arange(n_epoch):

            print("==========train===========")
            epoch_loss = 0
            correct_count=0
            training_loss=0
            start_time = time.time()
            shuffle(self.instances_train)
            instances_train = self.instances_train[:10]

            for i, instance in enumerate(instances_train):
                self.optim.zero_grad()

                outputs={}
                
                for choice_id in np.arange(4):
                    #if data_noise_percent=='0':
                    outputs[choice_id],_ = self.classifier(instance.question_text, instance.choices_text[choice_id], instance.science_fact_text, instance.knowledge_fact_text)
                    # elif data_noise_percent=='30':
                    #     outputs[choice_id] = self.classifier(instance.question_text, instance.choices_text[choice_id], instance.science_fact_text[choice_id], instance.knowledge_fact_text[choice_id])

                output = torch.cat([outputs[0], outputs[1], outputs[2], outputs[3]], dim=0).view(1, 4)
                answers = torch.argmax(torch.tensor(instance.target))

                loss, pred = self.classifier.get_loss(output, answers)
                if pred == answers:
                    correct_count += 1

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(classifier.parameters(), 20.0)


                self.optim.step()

                training_loss += loss
                # input('press enter to continue')
                if (i-1)%5==0:
                    print('processing sample ',i)
                if (i - 1) % 200 == 0 and i!=1:
                    model_results[epoch, 0]+=training_loss
                    print('sample ', i, 'average training loss:', training_loss / 200, '   200 sample time:', time.time()-start_time)
                    training_loss = 0
                    start_time = time.time()
                    
            model_results[epoch, 0]+=training_loss
            print('train acc:', correct_count / 1.0 / len(instances_train))
            model_results[epoch,0] = model_results[epoch,0]/len(instances_train)
            model_results[epoch, 1] = correct_count / 1.0 / len(instances_train)

    def train_interactive_forward(self, instance, choice_id):

        print("==========train interactive===========")
        self.optim.zero_grad()

        facts_text = list([])
        facts_text.extend([" ".join(single_fact) for single_fact in instance.knowledge_fact_text[-10:]])
        facts_text.append(" ".join(instance.science_fact_text))

        output, att_scores = self.classifier(instance.question_text, instance.choices_text[choice_id], instance.science_fact_text, instance.knowledge_fact_text)
        

        self.outputs[choice_id] = output
        if choice_id==np.argmax(np.array(instance.target)):
            self.att_scores = att_scores

        att_scores_list = [att_score_list.detach().tolist() for att_score_list in att_scores]
        return output.detach().tolist(), att_scores_list, " ".join(instance.question_text), " ".join(instance.choices_text[choice_id]), facts_text

    def train_interactive_backward(self, target, user_selection):
        model_outputs = torch.stack([self.outputs[0], self.outputs[1],self.outputs[2],self.outputs[3]])
        model_att_scores = self.att_scores

        loss, prediction = self.classifier.get_loss_interactive(model_outputs, target, model_att_scores, user_selection)
        loss.backward()
        self.optim.step()

        return loss, prediction

            # elif data_noise_percent=='30':
            #     outputs[choice_id] = self.classifier(instance.question_text, instance.choices_text[choice_id], instance.science_fact_text[choice_id], instance.knowledge_fact_text[choice_id])

    def test(self):

        print("==========dev===========")
        epoch_loss = 0
        correct_count = 0
        dev_loss = 0
        for i, instance in enumerate(self.instances_dev):
            self.optim.zero_grad()

            outputs = {}
            for choice_id in np.arange(4):
                #if data_noise_percent=='0':
                outputs[choice_id],_ = self.classifier(instance.question_text, instance.choices_text[choice_id], instance.science_fact_text, instance.knowledge_fact_text)
                #elif data_noise_percent=='30':
                #    outputs[choice_id] = self.classifier(instance.question_text, instance.choices_text[choice_id], instance.science_fact_text[choice_id], instance.knowledge_fact_text[choice_id])

            output = torch.cat([outputs[0], outputs[1], outputs[2], outputs[3]], dim=0).view(1, 4)
            answers = torch.argmax(torch.tensor(instance.target))

            loss, pred = self.classifier.get_loss(output, answers)
            if pred == answers:
                correct_count += 1

            dev_loss += loss
            # input('press enter to continue')
            if (i - 1) % 200 == 0 and i!=1:
                print('sample ', i, 'average training loss:', dev_loss / 200)
                dev_loss = 0
        print('dev acc:', correct_count / 1.0 / len(self.instances_dev))
        #model_results[epoch, 2] = dev_loss/len(self.instances_dev)
        #model_results[epoch, 3] = correct_count / 1.0 / len(self.instances_dev)







#save_folder_path = arch_name+'_noise_'+data_noise_percent+'_sciFact_'+n_sci_fact+'_knowFact_'+n_know_fact

    
    #torch.save(classifier, save_folder_path+'/MemNet_Trained_Epoch_' + str(epoch))

    

#np.savetxt(save_folder_path+'/acc.csv', model_results, delimiter=',')


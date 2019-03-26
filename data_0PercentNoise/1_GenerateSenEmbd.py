import numpy as np

import numpy as np
import json
import re
import collections
import pickle
import torch


unk_dict = {}
known_dict = {}

dict_path = "/Users/zhengzhongliang/PycharmProjects/DyMemNet/Glove_Embedding/glove.840B.300d.pickle"
VOC_DIM=300


with open(dict_path, 'rb') as input_file:
    glove_dict = pickle.load(input_file)

class Instance:
  def __init__(self, target,  question_text, choices_text, fact_text, knowledge_text):
    self.question_text = question_text
    self.choices_text = choices_text
    self.science_fact_text = fact_text
    self.knowledge_fact_text = knowledge_text
    self.target = target


def normalize_sen(sen):
  sen = sen.replace("?"," ? ")
  sen = sen.replace("."," . ")
  sen = sen.replace(","," , ")
  sen = sen.replace("!"," ! ")
  sen = sen.replace("'s"," 's ")
  sen = sen.replace("n't"," n't ")
  sen = sen.replace("-"," - ")
  sen = sen.replace(":"," : ")
  sen = sen.replace(";"," ; ")
  sen = sen.replace("%"," % ")
  sen = sen.replace("/"," / ")
  sen = sen.replace("("," ( ")
  sen = sen.replace(")"," ) ")
  sen = sen.replace("["," [ ")
  sen = sen.replace("]"," ] ")
  sen = sen.replace("'d"," 'd ")
  sen = sen.replace("'ll"," 'll ")
  sen = sen.replace("'ve"," 've ")

  word_list = list([])
  for word in sen.split():
    word = re.sub(r"'$", " ' ", word)
    word_list.append(word)

  return ' '.join([word for word in word_list])

# def sen_to_embd(sen, word2vec):
#     sen_mat = list([])
#     if len(sen)>0:
#       for word in sen.split():
#           if word in word2vec:
#               known_dict[word]=1
#               sen_mat.append(word2vec[word])
#           else:
#               unk_dict[word]=1
#               #sen_mat.append(np.ones(100)*0.1)
#               sen_mat.append(np.random.rand(VOC_DIM))
#       sen_mat = np.array(sen_mat)
#       #print(sen, sen_mat.shape)
#       slen, elen = sen_mat.shape
#       slen=max(2, slen)
#       l = [[(1 - s/(slen-1)) - (e/(elen-1)) * (1 - 2*s/(slen-1)) for e in range(elen)] for s in range(slen)]
#       l = np.array(l)
#       sen_vec = np.sum(np.multiply(sen_mat, l), axis=0)
#       return torch.tensor(sen_vec, dtype=torch.float32)

def output_questions(question_path, word2vec, output_file_name, data_portion):
  instance_list=list([])
  def normalize(raw_text):
    norm_text = [normalize_sen(sen) for sen in raw_text]
    return norm_text

  def get_common_knowledge(questionNum, data_portion):
    score_path = "/Users/zhengzhongliang/IdeaProjects/20190122_Test_2/RealRun_1/Result/"+data_portion+"/question_"+str(questionNum)+".txt_scores.tsv"


    with open(score_path,'r') as tsv:
      AoA = [line.strip().split('\t') for line in tsv]
    
    score_fact_map = np.zeros((len(AoA),2))
    for i, line_score in enumerate(AoA):
      score_fact_map[i,0] = line_score[0]
      score_fact_map[i,1] = line_score[2]  

    sorted_index = np.argsort(score_fact_map[:,0])
    selected_index = sorted_index[-50:]  # selected facts of top 10 scores
    selected_fact = score_fact_map[selected_index,1]

    question_path = "/Users/zhengzhongliang/IdeaProjects/20190122_Test_2/OpenBookQA-V1-Sep2018/Data/Additional/"+data_portion+"_complete.jsonl"
    fact_path = "/Users/zhengzhongliang/IdeaProjects/20190122_Test_2/OpenBookQA-V1-Sep2018/Data/Additional/crowdsourced-facts.txt"

    knowledge_text = list([])
    with open(fact_path, 'r') as the_file:
      #all_data = [line.strip() for line in the_file.readlines()]
      all_data = [line for line in the_file.readlines()]

      for j, factNum in enumerate(selected_fact):
        #print('factNum:',factNum, '    data length:',len(all_data))
        single_fact = all_data[int(factNum)].replace('\n', ' ')+' . '
        single_fact = normalize_sen(single_fact)
        knowledge_text.append(single_fact.split())
    #knowledge_text = ''.join(knowledge_text)
    #knowledge_text = knowledge_text.replace('\n', ' ')
    # print(knowledge_text)
    # input('press enter to continue')
    return knowledge_text

  with open(question_path, 'r',encoding='utf-8') as dataset:
    count=0
    for questionNum, line in enumerate(dataset):
      item_json = json.loads(line.strip())
      question_text = item_json["question"]["stem"]
      question_text = normalize_sen(question_text)
      question_text = question_text.split()

      #print('question text:', question_text)

      choices_text={}
      choice_count = 0
      for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
        sen_vec = list([])
        choice_text = choice_item["text"]
        choice_text = normalize_sen(choice_text)
        choices_text[choice_id] = choice_text.split()

        #print('choice text:', choices_text[choice_id])

      answer_label = item_json["answerKey"]
      if answer_label=='A':
        answer_list = [1,0,0,0]
      elif answer_label=='B':
        answer_list = [0,1,0,0]
      elif answer_label=='C':
        answer_list = [0,0,1,0]
      elif answer_label=='D':
        answer_list = [0,0,0,1]
      count+=1

      fact_text = item_json["fact1"]
      fact_text = normalize_sen(fact_text)
      fact_text = fact_text.split()

      #print('fact text:', fact_text)

      knowledge_text = get_common_knowledge(questionNum, data_portion)
      #knowledge_text = normalize_sen(knowledge_text)
      #knowledge_text = knowledge_text.split()

      #input('press enter to continue')

      instance_list.append(Instance(answer_list, question_text, choices_text, fact_text, knowledge_text))

  with open(output_file_name, 'wb') as handle:
    pickle.dump(instance_list, handle)

  return 0

# def open_text_to_mem(word2vec, opentext_path):
#   sen_list = list([])
#   with open(opentext_path,'r') as f:
#     for i, line in enumerate(f):
#       sen = line.replace('\"', ' ')
#       sen = sen.replace('\n', ' ')
#       sen_list.append(normalize_sen(sen))
#
#   mem_block = np.zeros((len(sen_list), VOC_DIM))
#   for i, sen in enumerate(sen_list):
#       mem_block[i] = sen_to_embd(sen, word2vec)
#   np.save('MemNet_MemoryBlock_Pos', mem_block)
#   return 0
#
# print('generating kb')
# open_text_to_mem(glove_dict,'OpenBookQA-V1-Sep2018/Data/Main/openbook.txt')

print('generating train')
train_list = output_questions('OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl', glove_dict,'MemNet_TrainQuestions_Pos.pickle','train')

print('generating dev')
dev_list = output_questions('OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl', glove_dict,'MemNet_DevQuestions_Pos.pickle','dev')

print('generating test')
test_list = output_questions('OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl', glove_dict,'MemNet_TestQuestions_Pos.pickle','test')


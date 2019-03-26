import numpy as np
import json
import re
import collections
import pickle
import torch

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

def output_facts(fact_path, output_filename):
  knowledge_text = list([])
  with open(fact_path, 'r') as the_file:
    #all_data = [line.strip() for line in the_file.readlines()]
    all_data = [line for line in the_file.readlines()]
  for i, line in enumerate(all_data):
    sen_norm = normalize_sen(line)
    if len(sen_norm)>0:
     knowledge_text.append(sen_norm.split())

  print('length of knowledge base:',len(knowledge_text))
  print('first instance in the knowledge base:', knowledge_text[0])

  with open(output_filename, 'wb') as handle:
    pickle.dump(knowledge_text, handle)

  return 0

fact_path_CK = "/Users/zhengzhongliang/IdeaProjects/20190228_Test/OpenBookQA-V1-Sep2018/Data/Additional/crowdsourced-facts.txt"
fact_path_SF = "/Users/zhengzhongliang/IdeaProjects/20190228_Test/OpenBookQA-V1-Sep2018/Data/Main/openbook.txt"


output_facts(fact_path_SF, 'science_fact.pickle')
output_facts(fact_path_CK, 'common_knowledge.pickle')
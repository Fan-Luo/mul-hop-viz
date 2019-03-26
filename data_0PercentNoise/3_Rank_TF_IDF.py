import numpy as np
import re
import pickle

class Instance:
  def __init__(self, question, choices, target, question_text, choices_text):
    self.question = question
    self.choices = choices
    self.target = target
    self.question_text = question_text
    self.choices_text = choices_text
    self.facts_rank = 0

def cosine_simi(A, B):
  nomi = np.dot(A,B)
  deno = np.linalg.norm(A)*np.linalg.norm(B)
  return nomi/deno

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

  sen=sen.lower()

  word_list = list([])
  for word in sen.split():
    word = re.sub(r"'$", " ' ", word)
    word_list.append(word)

  return ' '.join([word for word in word_list])

def build_index_tf_idf():

  # step 1: generate word space dict: loop over train, dev, test and corpus to generate dict
  print('step 1: generating word space dict')
  with open('MemNet_TrainQuestions_Pos.pickle', 'rb') as input_file:
    instances_train = pickle.load(input_file)

  with open('MemNet_DevQuestions_Pos.pickle', 'rb') as input_file:
    instances_dev = pickle.load(input_file)

  with open('MemNet_TestQuestions_Pos.pickle', 'rb') as input_file:
    instances_test = pickle.load(input_file)

  word_count = 0
  space_dict = {}
  for instance in instances_train:
    sen = instance.question_text
    for choice_text in instance.choices_text:
      sen = sen+' '+choice_text
    sen_norm = normalize_sen(sen)
    for word in sen_norm.split():
      if word not in space_dict:
        space_dict[word]=word_count
        word_count+=1

  for instance in instances_dev:
    sen = instance.question_text
    for choice_text in instance.choices_text:
      sen = sen+' '+choice_text
    sen_norm = normalize_sen(sen)
    for word in sen_norm.split():
      if word not in space_dict:
        space_dict[word]=word_count
        word_count+=1

  for instance in instances_test:
    sen = instance.question_text
    for choice_text in instance.choices_text:
      sen = sen+' '+choice_text
    sen_norm = normalize_sen(sen)
    for word in sen_norm.split():
      if word not in space_dict:
        space_dict[word]=word_count
        word_count+=1

  opentext_list = list([])
  with open('OpenBookQA-V1-Sep2018/Data/Main/openbook.txt','r') as f:
    for i, line in enumerate(f):
      sen = line.replace('\"', ' ')
      sen = sen.replace('\n', ' ')
      sen_norm = normalize_sen(sen)
      opentext_list.append(sen_norm)
      for word in sen_norm.split():
        if word not in space_dict:
          space_dict[word]=word_count
          word_count+=1

  # step 2: generate term frequency score for each doc:
  print('step 2: generating tf score for the opentext sentences')
  tf_score = np.zeros((len(opentext_list), len(space_dict)))
  for i, sen in enumerate(opentext_list):
    # print('========doc i:', i,'=========')
    for word in sen.split():
      # print('word:',word)
      tf_score[i, space_dict[word]]+=1
      # print('normed score:',score_normed)
      # input('press enter to continue')
    tf_score[i] = tf_score[i]/len(sen.split())

  # step 3: generate idf score:
  print('step 3: generating idf score for the opentext sentences')
  idf_count_mat = np.zeros((len(opentext_list), len(space_dict)))
  for word in space_dict:
    for i, sen in enumerate(opentext_list):
      if word in sen.split():
        idf_count_mat[i, space_dict[word]]=1
  idf_score_mat_ = np.sum(idf_count_mat, axis=0)

  idf_score = np.zeros(len(space_dict))
  for i in np.arange(len(space_dict)):
    if idf_score_mat_[i]!=0:
      idf_score[i] = np.log(len(opentext_list)/idf_score_mat_[i])

  # step 4: compute tf_itf score:
  print('step 4: generating tf-idf score for the opentext sentences')
  tf_idf_score = np.zeros((len(opentext_list), len(space_dict)))
  for i in np.arange(len(opentext_list)):
    tf_idf_score[i] = np.multiply(tf_score[i], idf_score)

  # step 5: generate customized memory for each question
  print('step 5: generating cosine score for each question')
  print('  generating for training data')
  for instance in instances_train:
    sen = instance.question_text
    for choice_text in instance.choices_text:
      sen = sen+' '+choice_text
    sen_norm = normalize_sen(sen)
    query = np.zeros(len(space_dict))
    for word in sen_norm.split():
      query[space_dict[word]]+=1
    cosine_score_all = np.zeros(len(opentext_list))
    for i in np.arange(len(opentext_list)):
      cosine_score = cosine_simi(tf_idf_score[i,:], query)
      cosine_score_all[i]=cosine_score
    instance.facts_rank = cosine_score_all
  with open('MemNet_TrainQuestions_Pos.pickle', 'wb') as handle:
    pickle.dump(instances_train, handle)

  print('  generating for dev data')
  for instance in instances_dev:
    sen = instance.question_text
    for choice_text in instance.choices_text:
      sen = sen+' '+choice_text
    sen_norm = normalize_sen(sen)
    query = np.zeros(len(space_dict))
    for word in sen_norm.split():
      query[space_dict[word]]+=1
    cosine_score_all = np.zeros(len(opentext_list))
    for i in np.arange(len(opentext_list)):
      cosine_score = cosine_simi(tf_idf_score[i,:], query)
      cosine_score_all[i]=cosine_score
    instance.facts_rank = cosine_score_all
  with open('MemNet_DevQuestions_Pos.pickle', 'wb') as handle:
    pickle.dump(instances_dev, handle)

  print('  generating for testing data')
  for instance in instances_test:
    sen = instance.question_text
    for choice_text in instance.choices_text:
      sen = sen+' '+choice_text
    sen_norm = normalize_sen(sen)
    query = np.zeros(len(space_dict))
    for word in sen_norm.split():
      query[space_dict[word]]+=1
    cosine_score_all = np.zeros(len(opentext_list))
    for i in np.arange(len(opentext_list)):
      cosine_score = cosine_simi(tf_idf_score[i,:], query)
      cosine_score_all[i]=cosine_score
    instance.facts_rank = cosine_score_all
  with open('MemNet_TestQuestions_Pos.pickle', 'wb') as handle:
    pickle.dump(instances_test, handle)


  return tf_idf_score

_=build_index_tf_idf()

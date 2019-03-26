import pickle
import numpy as np
import torch

class Instance:
  def __init__(self, question, choices, target, question_text, choices_text):
    self.question = question
    self.choices = choices
    self.target = target
    self.question_text = question_text
    self.choices_text = choices_text


def check_mem(mem_mat_np):

  nan_vec_count = 0
  nan_vec_indices = list([])

  inf_vec_count = 0
  inf_vec_indices = list([])

  zero_norm_count = 0
  zero_norm_indices = list([])

  for i, vec in enumerate(mem_mat_np):
    if np.sum(np.isnan(vec))>0:
      print('nan is detected in mem matrix!')
      print('sentence %d,   vector %d' % (i, vec))
      nan_vec_indices.append(i)
      nan_vec_count+=1

  for i, vec in enumerate(mem_mat_np):
    if np.sum(np.isinf(vec))>0:
      print('inf is detected in mem matrix!')
      print('sentence %d,   vector %d' % (i, vec))
      inf_vec_indices.append(i)
      inf_vec_count+=1

  for i, vec in enumerate(mem_mat_np):
    if np.linalg.norm(vec)<10e-7:
      print('zero norm vec is detected in mem matrix!')
      print('sentence %d,   vector %d' % (i, vec))
      zero_norm_indices.append(i)
      zero_norm_count+=1

  return nan_vec_indices, inf_vec_indices, zero_norm_indices

def check_question_and_choice(dataset):
  nan_vec_count = 0
  nan_vec_indices = list([])

  inf_vec_count = 0
  inf_vec_indices = list([])

  zero_norm_count = 0
  zero_norm_indices = list([])
  for i, instance in enumerate(dataset):
    for vec in instance.question:
      vec = vec.numpy()
      if np.sum(np.isnan(vec))>0:
        print('nan is detected in question!')
        print('instance:', i , '    vec:', vec)
        nan_vec_indices.append(i)
        nan_vec_count+=1

      if np.sum(np.isinf(vec))>0:
        print('inf is detected in question!')
        print('instance:', i , '    vec:', vec)
        inf_vec_indices.append(i)
        inf_vec_count+=1

      if np.linalg.norm(vec)<10e-7:
        print('zero norm vec is detected in question!')
        print('instance:', i , '    vec:', vec)
        zero_norm_indices.append(i)
        zero_norm_count+=1

    for j, choice in enumerate(instance.choices):
      for vec in choice:
        vec = vec.numpy()
        if np.sum(np.isnan(vec))>0:
          print('nan is detected in choice!')
          print('instance:', i , '    choice:',j,'    vec:', vec)
          nan_vec_indices.append(i)
          nan_vec_count+=1

        if np.sum(np.isinf(vec))>0:
          print('inf is detected in choice!')
          print('instance:', i , '    choice:',j,'    vec:', vec)
          inf_vec_indices.append(i)
          inf_vec_count+=1

        if np.linalg.norm(vec)<10e-7:
          print('zero norm vec is detected in choice!')
          print('instance:', i , '    choice:',j,'    vec:', vec)
          print('choice text:', instance.choices_text[j])
          zero_norm_indices.append(i)
          zero_norm_count+=1
  return nan_vec_indices, inf_vec_indices, zero_norm_indices


mem_mat_np = np.load('MemNet_MemoryBlock_Pos.npy')
nan_ind, inf_ind, zero_ind = check_mem(mem_mat_np)

with open('MemNet_TrainQuestions_Pos.pickle', 'rb') as input_file:
        instances_train = pickle.load(input_file)

with open('MemNet_DevQuestions_Pos.pickle', 'rb') as input_file:
    instances_dev = pickle.load(input_file)

with open('MemNet_TestQuestions_Pos.pickle', 'rb') as input_file:
    instances_test = pickle.load(input_file)

print('========training set==========')
nan_ind_train, inf_ind_train, zero_ind_train = check_question_and_choice(instances_train)
nan_ind_dev, inf_ind_dev, zero_ind_dev = check_question_and_choice(instances_dev)
nan_ind_test, inf_ind_test, zero_ind_test = check_question_and_choice(instances_test)


import pickle	

dict_path = "/Users/zhengzhongliang/PycharmProjects/DyMemNet/Glove_Embedding/glove.840B.300d.pickle"


with open(dict_path, 'rb') as input_file:
    glove_dict = pickle.load(input_file)


print("Train" in glove_dict)
print("Cat" in glove_dict)
print("Cats" in glove_dict)
print("cats" in glove_dict)
print("cat" in glove_dict)
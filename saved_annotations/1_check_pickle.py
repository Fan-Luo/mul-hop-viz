import pickle



with open('zhengzhongliang_2019-04-09_1603.pickle', 'rb') as f:
    pickle_file = pickle.load(f)
    print(pickle_file)
    pickle_file = pickle.load(f)
    print(pickle_file)
    pickle_file = pickle.load(f)
    print(pickle_file)
    


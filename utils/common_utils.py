import os
import pickle

def save_obj(obj, file_name, root='obj/'):
    if not os.path.exists(root):
        os.makedirs(root)
    with open(root+ file_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name, root='obj/'):
    with open(root + file_name + '.pkl', 'rb') as f:
        return pickle.load(f)
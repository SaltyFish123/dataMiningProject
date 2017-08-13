from find_features import *

def read_data(path):
    clf = []
    for i in range(len(path)):
        with open(path[i], 'rb') as f:
            clf.append(pickle.load(f))
        f.close()
    return clf


def extract_v(data):
    result = []
    for i in range(1, 6):
        result.append(data[data['vowels_amount'] == i])
    return result

def extract_i(data):
    result = []
    for i in range(1, 6):
        result.append(data[data['index'] == i])
    return result

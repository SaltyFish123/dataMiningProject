## import modules here 
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from helper import *

VOWEL = [ 'AA', 'AE', 'AH', 'AO','AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
COLUMNS = ['AA', 'AE', 'AH', 'AO','AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'index']

################# training #################

def train(data, classifier_file):# do not change the heading of the function
    #pass # **replace** this line with your code
    trainData = preprocessing(data)
    x = trainData[COLUMNS[:-1]]
    y = np.asarray(trainData[COLUMNS[-1]], dtype = 'int')
    gnb = DecisionTreeClassifier()
    gnb.fit(x, y)
    with open(classifier_file, 'wb') as f:
        pickle.dump(gnb, f)
    f.close()
    

################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    #pass # **replace** this line with your code
    with open(classifier_file, 'rb') as f:
        clf = pickle.load(f)
    testData = preprocessing(data)
    a = testData[COLUMNS[:-1]]
    prediction = clf.predict(a)
    f.close()
    return list(prediction)

# return a list of tuple (the amount of the vowel pronounce, the index of the primary press vowel)
def preprocessing(words):
    global VOWEL, COLUMNS
    df = pd.DataFrame(columns = COLUMNS)
    df_row = {}
    for word in words:
        index = 1
        df_row = df_row.fromkeys(COLUMNS, 0)
        word = word.upper()
        # cut the word
        pronounces = word.split(' ')
        pronounces[0] = pronounces[0].split(':')[1]
        for pronounce in pronounces:
            if (len(pronounce) != 0):
                if (pronounce[:2] in VOWEL):
                    df_row[pronounce[:2]] += index
                    if (pronounce[-1] == '1'):
                        df_row['index'] = index
                    index += 1
        df = df.append(df_row, ignore_index = True)
    return df

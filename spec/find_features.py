## import modules here 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from helper import *

path = ['D:/The 4th semester/dataMining/spec/asset/detail_preprocessing.data', 
       'D:/The 4th semester/dataMining/spec/asset/preprocessing.data']
train_file_path = 'D:/The 4th semester/dataMining/spec/asset/training_data.txt'
classifier_file_path = 'D:/The 4th semester/dataMining/spec/asset/classifier_file.dat'
test_file_path = 'D:/The 4th semester/dataMining/spec/asset/tiny_test.txt'
VOWEL = [ 'AA', 'AE', 'AH', 'AO','AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
CONSONANT = ['P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
#PENULTIMATE_WORDENDING = ['SION', 'SIONS', 'TIONS', 'TION', 'IC']
#ANTEPENULTIMATE_WORDENDING = ['TY', 'GY', 'PHY', 'ized', 'ize', 'ish', 'less', 'able']
FRONT_WORDENDING = ['AL', 'ION', 'IVE', 'IC', 'OUS']
REDUCED_WORDENDING = ['LY', 'RY', 'TY', 'IES', 'IED', 'ING', 'NESS', 'ISM']
#PRUNE_C = ['ES', 'S', 'ED']
COLUMNS = ['available_vowels_amount', 'front', 'index']
#COLUMNS = ['vowels_amount', 'penultimate_word_ending', 'antepenultimate_word_ending', 'index']
DETAIL_COLUMNS = ['word', 'vowels_amount', 'consonant_amount', 'index']

def kf_scores(data):
    COLUMNS = ['AA', 'AE', 'AH', 'AO','AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'index']
    trainData = simple_preprocessing(data)
    #For preprocessing and one_hot_preprocessing methods
    x = trainData[COLUMNS[:-1]]
    y = np.asarray(trainData[COLUMNS[-1]], dtype = 'int')
    #x, y = list(zip(*trainData))
    gnb = DecisionTreeRegressor()
    return cross_val_score(gnb, x, y, cv = 10)


################# training #################

def train(data, classifier_file):# do not change the heading of the function
    #pass # **replace** this line with your code
    x, y = preprocessing(data)
    gnb = MultinomialNB()
    gnb.fit(x, y)
    with open(classifier_file, 'wb') as f:
        pickle.dump(gnb, f)
    f.close()
    

################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    #pass # **replace** this line with your code
    with open(classifier_file, 'rb') as f:
        clf = pickle.load(f)
    a, b = preprocessing(data)
    prediction = clf.predict(a)
    f.close()
    return list(prediction)

# return a list of tuple (the amount of the vowel pronounce, the index of the primary press vowel)
# tested
def preprocessing(words):
    global VOWEL, PENULTIMATE_WORDENDING, ANTEPENULTIMATE_WORDENDING
    COLUMNS = ['word', 'vowels_amount', 'penultimate_word_ending', 'antepenultimate_word_ending', 'index']
    df = pd.DataFrame(columns = COLUMNS)
    df_row = {}
    for word in words:
        temp = ''
        df_row['vowels_amount'] = 0
        df_row['penultimate_word_ending'] = 0
        df_row['antepenultimate_word_ending'] = 0
        df_row['index'] = 0
        word = word.upper()
        # cut the word
        pronounces = word.split(' ')
        pronounces[0], temp = pronounces[0].split(':')
        pronounces.insert(1, temp)
        df_row['word'] = pronounces[0]
        for pronounce in pronounces[1:]:
            if (len(pronounce) != 0):
                if (pronounce[:2] in VOWEL):
                    df_row['vowels_amount'] += 1
                    if (pronounce[-1] == '1'):
                        df_row['index'] = df_row['vowels_amount']
        if (df_row['vowels_amount'] >= 3):
            if (pronounces[0][-2:] in ANTEPENULTIMATE_WORDENDING or pronounces[0][-3:] in ANTEPENULTIMATE_WORDENDING or pronounces[0][-4:] in ANTEPENULTIMATE_WORDENDING):
                df_row['antepenultimate_word_ending'] = 1
            if (pronounces[0][-2:] in PENULTIMATE_WORDENDING or pronounces[0][-4:] in PENULTIMATE_WORDENDING or pronounces[0][-5:] in PENULTIMATE_WORDENDING):
                df_row['penultimate_word_ending'] = 1
        df = df.append(df_row, ignore_index = True)
    return df
# tested but doesn't work well
def my_clf(data):
    target = []
    for i in range(data.shape[0]):
        if (data.iloc[i][COLUMNS[0]] < 3):
            target.append(1)
        elif(data.iloc[i][COLUMNS[1]] == 1):
            target.append(data.iloc[i][COLUMNS[0]] - 1)
        elif(data.iloc[i][COLUMNS[2]] == 1):
            target.append(data.iloc[i][COLUMNS[0]] - 2)
        else:
            continue
    return np.asarray(target)

def one_hot_preprocessing(words):
    global VOWEL, PENULTIMATE_WORDENDING, ANTEPENULTIMATE_WORDENDING, COLUMNS
    COLUMNS = ['one', 'two', 'three', 'four', 'five', 'penultimate_word_ending', 'antepenultimate_word_ending', 'index']
    df = pd.DataFrame(columns = COLUMNS)
    df_row = {}
    for word in words:
        vowels_amount = 0
        temp = ''
        df_row = df_row = df_row.fromkeys(COLUMNS, 0)
        word = word.upper()
        # cut the word
        pronounces = word.split(' ')
        pronounces[0], temp = pronounces[0].split(':')
        pronounces.insert(1, temp)
        for pronounce in pronounces[1:]:
            if (len(pronounce) != 0):
                if (pronounce[:2] in VOWEL):
                    vowels_amount += 1
                    if (pronounce[-1] == '1'):
                        df_row['index'] = vowels_amount
        if (vowels_amount >= 3):
            if (pronounces[0][-2:] in ANTEPENULTIMATE_WORDENDING or pronounces[0][-3:] in ANTEPENULTIMATE_WORDENDING):
                df_row['antepenultimate_word_ending'] = 1
            if (pronounces[0][-4:] in PENULTIMATE_WORDENDING or pronounces[0][-2:] in PENULTIMATE_WORDENDING):
                df_row['penultimate_word_ending'] = 1
        df_row[COLUMNS[vowels_amount - 1]] = 1
        df = df.append(df_row, ignore_index = True)
    return df

def size_preprocessing(words):
    result = []
    global VOWEL
    for word in words:
        vowelsAmount = [0]
        index = 0
        pronounces = word.split(' ')
        pronounces[0] = pronounces[0].split(':')[1]
        for pronounce in pronounces:
            if (len(pronounce) != 0):
                if (pronounce[:2] in VOWEL):
                    vowelsAmount[0] += 1
                    if (pronounce[-1] == '1'):
                        index = vowelsAmount[0]
        result.append((vowelsAmount.copy(), index))   
    return result

def size5_preprocessing(words):
    global VOWEL
    COLUMNS = ['one', 'two', 'three', 'four', 'five', 'index']
    df = pd.DataFrame(columns = COLUMNS)
    df_row = {}
    for word in words:
        temp = ''
        vowels_amount = 0
        df_row = df_row.fromkeys(COLUMNS, 0)
        word = word.upper()
        # cut the word
        pronounces = word.split(' ')
        pronounces[0], temp = pronounces[0].split(':')
        pronounces.insert(1, temp)
        for pronounce in pronounces[1:]:
            if (len(pronounce) != 0):
                if (pronounce[:2] in VOWEL):
                    vowels_amount += 1
                    if (pronounce[-1] == '1'):
                        df_row['index'] = vowels_amount
        df_row[COLUMNS[vowels_amount - 1]] = 1
        df = df.append(df_row, ignore_index = True)
    return df

def detail_preprocessing(words):
    global VOWEL, PENULTIMATE_WORDENDING, ANTEPENULTIMATE_WORDENDING, COLUMNS
    df = pd.DataFrame(columns = DETAIL_COLUMNS)
    df_row = {}
    for word in words:
        temp = ''
        df_row['vowels_amount'] = 0
        df_row['index'] = 0
        df_row['consonant_amount'] = 0
        word = word.upper()
        # cut the word
        pronounces = word.split(' ')
        pronounces[0], temp = pronounces[0].split(':')
        pronounces.insert(1, temp)
        df_row['word'] = pronounces[0]
        for pronounce in pronounces[1:]:
            if (len(pronounce) != 0):
                if (pronounce[:2] in VOWEL):
                    df_row['vowels_amount'] += 1
                    if (pronounce[-1] == '1'):
                        df_row['index'] = df_row['vowels_amount']
                else:
                    df_row['consonant_amount'] += 1
        df = df.append(df_row, ignore_index = True)
    return df

def vowel_preprocessing(words):
    global VOWEL, PENULTIMATE_WORDENDING, ANTEPENULTIMATE_WORDENDING
    COLUMNS = ['word', 'AA', 'AE', 'AH', 'AO','AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'index']
    df = pd.DataFrame(columns = COLUMNS)
    df_row = {}
    for word in words:
        temp = ''
        index = 0
        df_row = df_row.fromkeys(COLUMNS, 0)
        word = word.upper()
        # cut the word
        pronounces = word.split(' ')
        pronounces[0], temp = pronounces[0].split(':')
        pronounces.insert(1, temp)
        df_row['word'] = pronounces[0]
        for pronounce in pronounces[1:]:
            if (len(pronounce) != 0):
                if (pronounce[:2] in VOWEL):
                    df_row[pronounce[:2]] += 1
                    index += 1
                    if (pronounce[-1] == '1'):
                        df_row['index'] = index
        df = df.append(df_row, ignore_index = True)
    return df

def reducing_preprocessing(words):
    global VOWEL, REDUCED_WORDENDING, FRONT_WORDENDING, SECOND_WORDBEGINING, SECOND_WORDENDING, COLUMNS
    df = pd.DataFrame(columns = COLUMNS)
    stress = list('012')
    df_row = {}
    for word in words:
        content = ''
        vowels_index = []
        prune = False
        df_row = df_row.fromkeys(COLUMNS, 0)
        word = word.upper()
        # cut the word
        pronounces = word.split(' ')
        content, pronounces[0] = pronounces[0].split(':')

        for pronounce in pronounces:
            if (len(pronounce) != 0):
                if (pronounce[:2] in VOWEL):
                    df_row['available_vowels_amount'] += 1
                    if (pronounce[-1] in stress):
                        if (pronounce[-1] == '1'):
                            df_row['index'] = df_row['available_vowels_amount']
                            prune = True
                    pronounce = pronounce[:-1]
        '''
        for i in range(len(pronounces)):
            if (len(pronounces[i]) != 0):
                if (pronounces[i][:2] in VOWEL):
                    df_row['available_vowels_amount'] += 1
                    if (pronounces[i][-1] in stress):
                        vowels_index.append(i)
                        if (pronounces[i][-1] == '1'):
                            df_row['index'] = df_row['available_vowels_amount']
                            prune = True
                    pronounces[i] = pronounces[i][:-1]
        '''        
        if (prune and df_row['available_vowels_amount'] >= 3):
            for i in REDUCED_WORDENDING:
                if (i in content[-len(i) - 2:]):
                    df_row['available_vowels_amount'] -= 1
                    content = content[:content.find(i)]
                    break
            for i in FRONT_WORDENDING:
                if (i in content[-len(i) - 2:]):
                    df_row['front'] = 1
                    break
        df = df.append(df_row, ignore_index = True)
    return df

def simple_preprocessing(words):
    global VOWEL
    COLUMNS = ['AA', 'AE', 'AH', 'AO','AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'index']
    df = pd.DataFrame(columns = COLUMNS)
    df_row = {}
    for word in words:
        temp = ''
        index = 1
        df_row = df_row.fromkeys(COLUMNS, 0)
        word = word.upper()
        # cut the word
        pronounces = word.split(' ')
        pronounces[0], temp = pronounces[0].split(':')
        pronounces.insert(1, temp)
        for pronounce in pronounces[1:]:
            if (len(pronounce) != 0):
                if (pronounce[:2] in VOWEL):
                    df_row[pronounce[:2]] += index
                    if (pronounce[-1] == '1'):
                        df_row['index'] = index
                    index += 1
        df = df.append(df_row, ignore_index = True)
    return df

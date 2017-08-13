from  helper import *
from submission import *

train_file_path = 'D:/The 4th semester/dataMining/spec/asset/training_data.txt'
classifier_file_path = 'D:/The 4th semester/dataMining/spec/asset/classifier_file.dat'
test_file_path = 'D:/The 4th semester/dataMining/spec/asset/tiny_test.txt'

training_data = read_data(train_file_path)
train(training_data, classifier_file_path)
testing_data = read_data(test_file_path)
prediction = test(testing_data, classifier_file_path)
print(prediction)

'''
training_data = read_data(train_file_path)
print(kf_scores(training_data))
'''
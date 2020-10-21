"""
Error occured on generating json.

Let's make my own dataloader.
To this end, read CSV and count segments are needed beforehand.
"""


path = '../../../../raid/Hollywood2/Hollywood2/ClipSets/'
frame_path = '../../../../raid/Hollywood2/cut/'

import csv 
import os

class_list = ['AnswerPhone',
            'DriveCar',
            'Eat',
            'FightPerson',
            'GetOutCar',
            'HandShake',
            'HugPerson',
            'Kiss',
            'Run',
            'SitDown',
            'SitUp',
            'StandUp']

# train
train_data = []

for c in class_list:
    with open(path + c + '_train.txt', newline='') as f:
        tmp = f.readline()
        while tmp:
            tmp = tmp.split(' ')
            if tmp[2][0] == '1':
                count = len(os.listdir(frame_path + tmp[0]))
                train_data.append([tmp[0], c, count])
            #next line
            tmp = f.readline()

with open('holly_trainingSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',')
    for i in range(len(train_data)):
        train_writer.writerow(train_data[i])

# validation
train_data = []

for c in class_list:
    with open(path + c + '_test.txt', newline='') as f:
        tmp = f.readline()
        while tmp:
            tmp = tmp.split(' ')
            if tmp[2][0] == '1':
                count = len(os.listdir(frame_path + tmp[0]))
                train_data.append([tmp[0], c, count])
            #next line
            tmp = f.readline()

with open('holly_validationSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',')
    for i in range(len(train_data)):
        train_writer.writerow(train_data[i])


# inference
# example : {'actioncliptest00165': 'DriveCar|Run'}
test_data = {}
for c in class_list:
    with open(path + c + '_test.txt', newline='') as f:
        tmp = f.readline()
        while tmp:
            tmp = tmp.split(' ')
            if tmp[2][0] == '1':
                if tmp[0] in test_data:
                    test_data[tmp[0]] = test_data[tmp[0]] + '|' + c
                else:
                    test_data[tmp[0]] = c
            #next line
            tmp = f.readline()
with open('holly_inferenceSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',')
    for k, v in test_data.items():
        count = len(os.listdir(frame_path + k))
        train_writer.writerow([k, v, count])

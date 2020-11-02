"""
Error occured on generating json.

Let's make my own dataloader.
To this end, read CSV and count segments are needed beforehand.
"""


path = '../../../../raid/jester/'

import csv 
import os

# row : ['34870', 'Drumming Fingers']
train_data = []

with open(path + 'jester-v1-train.csv', newline='') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in train_reader:
        train_data.append(row)

with open('jester_trainingSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(train_data)):
        name = train_data[i][0]
        tmp = os.listdir(path + '20bn-jester-v1/' + name)
        train_writer.writerow([name, train_data[i][1], len(tmp)])

train_data = []
with open(path + 'jester-v1-validation.csv', newline='') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in train_reader:
        train_data.append(row)

with open('jester_validationSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(train_data)):
        name = train_data[i][0]
        tmp = os.listdir(path + '20bn-jester-v1/' + name)
        train_writer.writerow([name, train_data[i][1], len(tmp)])


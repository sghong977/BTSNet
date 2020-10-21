"""
Error occured on generating json.

Let's make my own dataloader.
To this end, read CSV and count segments are needed beforehand.
"""


path = '../../../../raid/SVW/'

import csv 
import os

# row : ['FileName', 'Genre', 'Train 1?']
train_data = []
test_data = []
with open(path + 'SVW.csv', newline='') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in train_reader:
        tmp = [row[0]] + [row[1].lower()]
        if row[11] == '1':
            train_data.append(tmp)
        if row[11] == '0':
            test_data.append(tmp)

with open('SVW_trainingSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(1, len(train_data)):
        name = train_data[i][0][0:-4]
        tmp = os.listdir(path+'cut/'+ train_data[i][1] +'/'+name)
        train_writer.writerow([name, train_data[i][1], len(tmp)])

with open('SVW_validationSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(1, len(test_data)):
        name = test_data[i][0][0:-4]
        tmp = os.listdir(path+'cut/'+ test_data[i][1] +'/'+name)
        train_writer.writerow([name, test_data[i][1], len(tmp)])

"""
# validation data
val_data = []
with open(path + 'validationSet.csv', newline='') as csvfile:
    val_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in val_reader:
        #print(row)
        val_data.append(row[0:2])

with open('new_validationSet.csv', 'w', newline='') as csvfile:
    val_writer = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(val_data)):
        name = val_data[i][0][0:-4]
        tmp = os.listdir(path+'validation/'+ name)
        val_writer.writerow([name.split('/')[1], val_data[i][1], len(tmp)])

"""
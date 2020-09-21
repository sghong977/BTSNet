"""
Error occured on generating json.

Let's make my own dataloader.
To this end, read CSV and count segments are needed beforehand.
"""


path = '../../../../raid/Moment/Moments_in_Time_256x256_30fps/'

import csv 
import os

# row : ['blocking/getty-karate-video-id635808620_4.mp4', 'blocking', '3', '0']
train_data = []
with open(path + 'trainingSet.csv', newline='') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in train_reader:
        #print(row)
        train_data.append(row[0:2])

with open('new_trainingSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(train_data)):
        name = train_data[i][0][0:-4]
        tmp = os.listdir(path+'training/'+ name)
        train_writer.writerow([name.split('/')[1], train_data[i][1], len(tmp)])
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
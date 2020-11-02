"""
Error occured on generating json.

Let's make my own dataloader.
To this end, read CSV and count segments are needed beforehand.
"""


path = '../../../../raid/Charades/'

import math
import csv 
import os
import numpy as np


# create mapping dictionary
num2ClassName = {}
with open(path + 'Charades_v1_classes.txt', newline='') as f:
    tmp = f.readline()
    while tmp:
        num2ClassName[tmp[0:4]] = tmp[5:-1]
        tmp = f.readline()

train_data = []
# need to convert sec to frame number (24 fps)
with open(path + 'Charades_v1_train.csv', newline='') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in train_reader:
        if row[0] == 'id':
            continue
        #eg. 46GP8 ['c092 11.90 21.20', 'c147 0.00 12.60'] 597
        id = row[0]
        classes = row[-2].split(';')
        fnum = len(os.listdir(path + 'cut/' + id))

        # extract range
        for clip in classes:
            if len(clip) == 0: continue
            clip = clip.split(' ')
            start = math.floor(float(clip[1]) * 24)
            end = math.floor(float(clip[2]) * 24)

            # condition 1 : end should be less than fnum (remove the clip)
            if end >= fnum:
                end = fnum
            # condition 2 : remove the clip if len is less than 30
            if (end - start) <= 30:
                continue
            if start >= fnum:
                continue
            train_data.append([id, num2ClassName[clip[0]], start, end-1])       

with open('charades_trainingSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(train_data)):
        train_writer.writerow(train_data[i])

# validation
train_data = []
# need to convert sec to frame number (24 fps)
with open(path + 'Charades_v1_test.csv', newline='') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in train_reader:
        if row[0] == 'id':
            continue
        id = row[0]
        classes = row[-2].split(';')
        fnum = len(os.listdir(path + 'cut/' + id))

        # extract range
        for clip in classes:
            if len(clip) == 0: continue
            clip = clip.split(' ')
            start = math.floor(float(clip[1]) * 24)
            end = math.floor(float(clip[2]) * 24)

            # condition 1 : end should be less than fnum (remove the clip)
            if end >= fnum:
                end = fnum
            # condition 2 : remove the clip if len is less than 30
            if (end - start) <= 30:
                continue
            if start >= fnum:
                continue
            train_data.append([id, num2ClassName[clip[0]], start, end-1])       

with open('charades_validationSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(train_data)):
        train_writer.writerow(train_data[i])


# inference
train_data = []
# need to convert sec to frame number (24 fps)
with open(path + 'Charades_v1_test.csv', newline='') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in train_reader:
        if row[0] == 'id':
            continue
        id = row[0]
        classes = ''

        # get class names
        clips = row[-2].split(';')

        # remove some empty cases
        if clips[0] == '':
            continue
        
        # add to dataset
        for clip in clips:
            if len(clip) == 0: continue 
            if classes == '':
                classes = num2ClassName[clip[0:4]]
            else:
                classes = classes + '|'+ num2ClassName[clip[0:4]]
        train_data.append([id, classes])       

with open('charades_inferenceSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',')
    for k, v in train_data:
        count = len(os.listdir(path + 'cut/' + k))
        # Inference Batch size = 25
        ranges = np.linspace(0,count,27)
        ranges = [int(i) for i in ranges]

        for r in range(len(ranges)-2):
            train_writer.writerow([k, v, ranges[r], ranges[r+2]])

"""
# need to convert sec to frame number (24 fps)
with open(path + 'Charades_v1_test.csv', newline='') as csvfile:
    train_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in train_reader:
        if row[0] == 'id':
            continue
        id = row[0]
        classes = ''
        fnum = len(os.listdir(path + 'cut/' + id))

        # get class names
        clips = row[-2].split(';')
        for clip in clips:
            if len(clip) == 0: continue 
            if classes == '':
                classes = num2ClassName[clip[0:4]]
            else:
                classes = classes + '|'+ num2ClassName[clip[0:4]]
        train_data.append([id, classes, fnum])       

with open('charades_inferenceSet.csv', 'w', newline='') as csvfile:
    train_writer = csv.writer(csvfile, delimiter=',')
    for i in range(len(train_data)):
        train_writer.writerow(train_data[i])
"""
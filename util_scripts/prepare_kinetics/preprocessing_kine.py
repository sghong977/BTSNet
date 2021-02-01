import numpy as np
import os
import glob
import csv
import shutil

CSV_PATH = '/nas/Public/Kinetics_700/annotations/kinetics_700_'
data_type = ['train', 'val', 'test']

DATA_PATH = '../../../raid/Kinetics700/'


#label,youtube_id,time_start,time_end,split
def create_label_file():
    labels = dict()

    with open(CSV_PATH + data_type[1] + '.csv', newline='') as csvfile:
        train_reader = csv.reader(csvfile, delimiter=',')
        for row in train_reader:
            if row[0] == 'label':
                continue
            labels[row[0]] = 0

    with open('./csv_and_json/kinetics700_labels.txt', 'w') as f:
        for i in labels:
            f.write(i+'\n')

def load_label_list():
    labels = list()
    with open('./csv_and_json/kinetics700_labels.txt', 'r') as f:
        while True:
            lb = f.readline()
            if not lb:
                break
            labels.append(lb[:-1])
            
    return labels

def create_annotation_file():
    global data_type, DATA_PATH
    labels = load_label_list()
    
    for t in ['train', 'validation']:
        with open('./csv_and_json/kinetics700_'+t+'.txt', 'w') as f:
            for l in labels:
                ids = os.listdir(DATA_PATH+t+'/'+l)
                for id in ids:
                    #name = id[:11]
                    f.write(id+','+l+'\n')

create_annotation_file()
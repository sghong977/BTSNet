import os
import json
import csv
import pandas as pd
import numpy as np
if __name__ == '__main__':
    # annotation_path = '/raid/video_data/epic/annotation/EPIC_train_action_labels.csv'
    # data = pd.read_csv(annotation_path)
    # class_noum = np.zeros((352))
    # for i in range(data.shape[0]):
    #     class_noum[data['noun_class'][i]] += 1
    # print(i)
    # # for i in class_noum:
    # #     print(i)
    

    annotation_path = '/raid/video_data/epic/annotation/val.csv'
    f = open(annotation_path, 'r')
    data = csv.reader(f)
    class_noum = np.zeros((352))
    cnt = 0
    for row in data:
        class_noum[int(row[1])] += 1
        cnt += 1
    for i in class_noum:
        print(i)
    print(cnt) 
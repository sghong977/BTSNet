import argparse
import json
from pathlib import Path
import os
import pandas as pd
import csv
from utils import epic_get_n_frames, epic_get_n_frames_flow

def s1_s2_convert_csv_to_dict(csv_path, subset, output_type, modality):
    data = pd.read_csv(csv_path)
    keys = []
    key_labels = []
    video_path = []

    for i in range(data.shape[0]):

        row = data.iloc[i, :]
        basename = str(row['uid'])
        keys.append(basename)
        if modality == 'flow':
            video_path.append(os.path.join('flow', row['participant_id'], row['video_id'], str(row['uid'])))
        elif modality == 'rgb':
            video_path.append(os.path.join('slowfast_frames', row['participant_id'], row['video_id'], str(row['uid'])))

    database = {}   
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        database[key]['video_path'] = video_path[i]
        database[key]['annotations'] = {}

    return database

def Fulltraining_convert_csv_to_dict(csv_path, subset, output_type, modality):
    data = pd.read_csv(csv_path)
    keys = []
    key_labels = []
    video_path = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]

        basename = str(row['uid'])
        keys.append(basename)
        if modality == 'flow':
            video_path.append(os.path.join('flow', row['participant_id'], row['video_id'], str(row['uid'])))
        elif modality == 'rgb':
            video_path.append(os.path.join('slowfast_frames', row['participant_id'], row['video_id'], str(row['uid'])))
        if output_type == 'noun':
            key_labels.append(str(row['noun_class']))
        elif output_type == 'verb':
            key_labels.append(str(row['verb_class']))

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        database[key]['video_path'] = video_path[i]
        if subset != 'testing':
            label = key_labels[i]
            database[key]['annotations'] = {'label': label}
        else:
            database[key]['annotations'] = {}

    return database

def load_labels(train_csv_path):
    data = pd.read_csv(train_csv_path)
    return data['label'].unique().tolist()


def convert_kinetics_csv_to_json(train_csv_path, output_type, dst_json_path, train_test, modality):
    if train_test == 'train':
        train_database = Fulltraining_convert_csv_to_dict(train_csv_path, 'training', output_type, modality)
    elif train_test == 'test':
        train_database = s1_s2_convert_csv_to_dict(train_csv_path, output_type, output_type, modality)
   
    dst_data = {}

    dst_data['database'] = {}
    dst_data['database'].update(train_database)


    for k, v in dst_data['database'].items():
     
        video_path = Path(v['video_path'])
        if Path('/raid/video_data/epic' / video_path).exists():
            #/raid/video_data/epic/slowfast_frames/P10/P10_04/16403
            n_frames = epic_get_n_frames('/raid/video_data/epic' / video_path)
            if modality == 'flow':
                n_frames = epic_get_n_frames('/raid/video_data/epic' / video_path / 'u')
            v['annotations']['segment'] = (0, n_frames)

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
 
    parser.add_argument('--train_test',
                        default='train',
                        type=str,
                        help=('train or test'))
    parser.add_argument('--modality',
                        default='rgb',
                        type=str,
                        help=('rgb or flow'))


    args = parser.parse_args()
    for li in ['train', 'test']:
        args.train_test = li
        if args.train_test == 'train':
            output_type_list = ['noun', 'verb']            
        else:

            output_type_list = ['s1', 's2']
        for output_type in output_type_list:
            dir_path = Path("/raid/video_data/epic/annotation/")
            video_path = Path("/raid/video_data/epic/slowfast_frames/")
    
            output_pth = '../csv_and_json/'
            if args.train_test == 'train':
                train_csv_path = '/raid/video_data/epic/annotation/EPIC_train_action_labels.csv'
                dst_path = Path(output_pth + '/epicfull_'+ args.modality +'_'+ output_type +'.json')
            elif args.train_test == 'test':
                train_csv_path = '/raid/video_data/epic/annotation/EPIC_test_'+ output_type +'_timestamps.csv'
                dst_path = Path(output_pth + '/epic_'+ args.modality +'_'+ output_type +'.json')
            

            convert_kinetics_csv_to_json(train_csv_path, output_type, dst_path, args.train_test, args.modality)

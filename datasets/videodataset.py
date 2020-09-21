import json
import csv
from pathlib import Path

import torch
import torch.utils.data as data

from .loader import VideoLoader


def get_class_labels(data, data_name, root_path):
    class_labels_map = {}
    
    if data_name == 'mit':
        f = open(root_path / 'moments_categories.txt', 'r')
        while True:
            tmp = f.readline()
            if not tmp: break
            tmp = tmp.split(',')
            class_labels_map[tmp[0]] = int(tmp[1])
    elif data_name == 'jester':
        f = open(root_path / 'jester-v1-labels.csv', 'r')
        i = 0
        while True:
            tmp = f.readline()
            if not tmp: break
            class_labels_map[tmp[:-1]] = i    # remove '\n'
            i += 1
    # for other datasets
    else:
        index = 0
        for class_label in data['labels']:
            class_labels_map[class_label] = index
            index += 1
    return class_labels_map

# root_path == opt.video_path
def get_database(data, subset, root_path, video_path_formatter, data_name):
    video_ids = []
    video_paths = []
    annotations = []
    segments = []

    # read new_trainingSet.csv
    if data_name == "mit":
        with open('new_'+ subset +'Set.csv', newline='') as csvfile:
            train_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in train_reader:
                # key
                video_ids.append(row[0])  # only file name
                video_paths.append(root_path / subset / row[1] / row[0])
                annotations.append(row[1])
                segments.append(int(row[2]))
        # debug
        #print(video_ids[0], video_paths[0], annotations[0])
    elif data_name == 'jester':
        with open('jester_'+ subset +'Set.csv', newline='') as csvfile:
            train_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in train_reader:
                # key
                video_ids.append(row[0])  # only file name
                video_paths.append(root_path / '20bn-jester-v1' / row[0])
                annotations.append(row[1])
                segments.append(int(row[2]))
    else:
        segments = None
        for key, value in data['database'].items():
            this_subset = value['subset']
            if this_subset == subset:
                video_ids.append(key)
                annotations.append(value['annotations'])
                if 'video_path' in value:
                    video_paths.append(Path(value['video_path']))
                else:
                    label = value['annotations']['label']
                    video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations, segments


class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 data_name=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):

        self.data_name = data_name

        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        # if 'mit', 'data' doesnt exist
        data = None
        if self.data_name not in ['mit', 'jester']:
            with annotation_path.open('r') as f:
                data = json.load(f)
        video_ids, video_paths, annotations, segments = get_database(
            data, subset, root_path, video_path_formatter, self.data_name)
        
        # redefine 'get_class_labels' for mit
        class_to_idx = get_class_labels(data, self.data_name, root_path)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            #no 'label' in mit.
            # we don't habve 'test'set in Mit dataset.
            if self.data_name == 'mit':
                label = annotations[i]
                label_id = class_to_idx[label]
                # segment : 
                segment = [0, segments[i]-1]
                if segment[1] == 1:
                    continue
                frame_indices = list(range(0, segments[i]))
            elif self.data_name == 'jester':
                label = annotations[i]
                label_id = class_to_idx[label]
                # segment : 
                segment = [1, segments[i]]
                if segment[1] == 1:
                    continue
                frame_indices = list(range(1, segments[i]+1))  #?

            #------- other datasets ---------------
            else:
                if 'label' in annotations[i]:
                    label = annotations[i]['label']   # annotations[i]
                    label_id = class_to_idx[label]
                else:
                    label = 'test'
                    label_id = -1
                segment = annotations[i]['segment']
                if segment[1] == 1:
                    continue

                frame_indices = list(range(segment[0], segment[1]))

            video_path = video_paths[i]
            if not video_path.exists():
                continue

            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
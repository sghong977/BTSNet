import json
import csv
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np
import math

from .loader import VideoLoader


def get_class_labels(data, data_name, root_path):
    class_labels_map = {}
    
    f = open(root_path / 'Charades_v1_classes.txt', 'r')
    i = 0
    while True:
        tmp = f.readline()
        if not tmp: break
        class_labels_map[tmp[5:-1]] = i    # remove '\n' and 'cxxx '
        i += 1
    return class_labels_map

# root_path == opt.video_path
def get_database(data, subset, root_path, video_path_formatter, data_name):
    video_ids = []
    video_paths = []
    annotations = []
    segments = []

    with open('charades_'+ subset +'Set.csv', newline='') as csvfile:
        train_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in train_reader:
            # key
            video_ids.append(row[0])  # only file name
            video_paths.append(root_path / 'cut' / row[0])
            annotations.append(row[1])
            segments.append([int(row[2]), int(row[3])])
    return video_ids, video_paths, annotations, segments

class Charades(data.Dataset):

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
                 target_type='label'  #
                ):
        self.target_type = target_type   #
        self.subset = subset

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


    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        # if 'mit', 'data' doesnt exist
        data = None
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

            # for charades && train/validation (except for inference)
            # same as above one on inference scenario
            if subset is not 'inference':
                label = annotations[i]
                label_id = class_to_idx[label]
                segment = segments[i]
                frame_indices = list(range(segment[0], segment[1]))
            else:
                label = annotations[i]
                # inference
                label_list = label.split('|')
                label_id = [class_to_idx[i] for i in label_list]
                segment = segments[i]
                tt = np.linspace(segment[0], segment[1], 30)
                frame_indices = [math.floor(i) for i in tt]

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
        # need to change here
        target = self.data[index]['label']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)

#------------------------- For validation and Inference

import json
import copy
import functools

import torch
from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    target_element = batch_targets[0]
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets


class CharadesMultiClips(Charades):

    def __loading(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
            segments.append(
                [min(clip_frame_indices),
                 max(clip_frame_indices) + 1])

        return clips, segments

    def __getitem__(self, index):
        path = self.data[index]['video']

        video_frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)
        clips, segments = self.__loading(path, video_frame_indices)

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        if 'segment' in self.target_type:
            if isinstance(self.target_type, list):
                segment_index = self.target_type.index('segment')
                targets = []
                for s in segments:
                    targets.append(copy.deepcopy(target))
                    targets[-1][segment_index] = s
            else:
                targets = segments
        else:
            targets = [target for _ in range(len(segments))]

        return clips, targets
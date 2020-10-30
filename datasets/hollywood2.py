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

    f = open(root_path / 'list.txt', 'r')
    i = 0
    while True:
        tmp = f.readline()
        if not tmp: break
        class_labels_map[tmp[:-1]] = i
        i += 1
    return class_labels_map

# root_path == opt.video_path
def get_database(data, subset, root_path, annotation_path, video_path_formatter, data_name):
    video_ids = []
    video_paths = []
    annotations = []
    segments = []
    
    #
    with open(str(annotation_path) + '/holly_'+ subset +'Set.csv', newline='') as csvfile:
        train_reader = csv.reader(csvfile, delimiter=',')
        for row in train_reader:
            # key
            video_ids.append(row[0])  # only file name
            video_paths.append(root_path / 'cut' / row[0])
            annotations.append(row[1])
            if subset == 'inference':
                segments.append([int(row[2]), int(row[3])])
            else:
                segments.append(int(row[2])) 

    return video_ids, video_paths, annotations, segments

class Hollywood2(data.Dataset):

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
                 target_type = 'label'):
        self.subset = subset
        self.target_type = target_type

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
            data, subset, root_path, annotation_path, video_path_formatter, self.data_name)
        
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

            label = annotations[i]
            # inference
            if subset is 'inference':
                label_list = label.split('|')
                label_id = [class_to_idx[i] for i in label_list]
                segment = segments[i]
                tt = np.linspace(segment[0], segment[1]-1, 30)    #
                frame_indices = [math.floor(i) for i in tt]
            # train/validation
            else:
                label_id = class_to_idx[label]
                segment = [0, segments[i]-1]
                if segment[1] == 1:
                    continue
                frame_indices = list(range(0, segments[i]))

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
        # change target when inference
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

# for inference
def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    #batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    #batch_targets = [
    #    target for multi_targets in batch_targets for target in multi_targets
    #]
    
    #print(len(batch_clips), batch_clips[0].shape)
    target_element = batch_targets[0]
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets

def collate_fn_val(batch):
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

class Hollywood2MultiClips(Hollywood2):

    def __loading__one(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

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

        if self.subset=='inference':
            if self.temporal_transform is not None:
                video_frame_indices = self.temporal_transform(video_frame_indices)
            clips = self.__loading__one(path, video_frame_indices)
        else:
            if self.temporal_transform is not None:
                video_frame_indices = self.temporal_transform(video_frame_indices)
            clips, segments = self.__loading(path, video_frame_indices)

        if isinstance(self.target_type, list):
            targets = [self.data[index][t] for t in self.target_type]
        else:
            targets = self.data[index][self.target_type]
        
        if 'segment' in self.target_type:
            if isinstance(self.target_type, list):
                segment_index = self.target_type.index('segment')
                targets = []
                for s in segments:
                    targets.append(copy.deepcopy(target))
                    targets[-1][segment_index] = s
            else:
                targets = segments
        # if inference, don't need eo expand to segments size
        elif self.subset == 'inference':
                pass
        else:
            targets = [targets for _ in range(len(segments))]
        
        return clips, targets
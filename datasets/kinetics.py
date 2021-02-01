import json
from pathlib import Path

import torch
import torch.utils.data as data

import numpy as np

from .loader import VideoLoader, VideoCutLoader

def get_class_labels():
    path = './csv_and_json/kinetics700_labels.txt'
    class_labels_map = {}
    f = open(path, 'r')
    i = 0
    while True:
        tmp = f.readline()
        if not tmp: break
        class_labels_map[tmp[:-1]] = i
        i += 1
    return class_labels_map

def get_database(subset, root_path):
    video_paths = []
    annotations = []

    path = './csv_and_json/kinetics700_' + subset + '.txt'
    f = open(path, 'r')
    while True:
        tmp = f.readline()
        if not tmp: break
        tmp = tmp.split(',')
        annotations.append(tmp[1][:-1])
        video_paths.append(root_path / subset / tmp[1][:-1] / tmp[0])
    return video_paths, annotations


class Kinetics700(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 data_name,

                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'frame{x:010d}.jpg',
                 target_type='label'):

        self.data, self.class_names = self.__make_dataset(root_path, subset)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.loader = VideoCutLoader()              # new loader

        self.target_type = target_type

    def __make_dataset(self, root_path, subset):
        video_paths, annotations = get_database(subset, root_path)
        class_to_idx = get_class_labels()
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[str(label)] = name

        n_videos = len(video_paths)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, n_videos))

            # test is not implemented yet
            label_id = class_to_idx[annotations[i]]
            video_path = video_paths[i]
            #if not video_path.exists():
            #    continue

            sample = {
                'video': video_path,   # full path
                #'segment': segment,
                #'frame_indices': frame_indices,
                #'video_id': video_ids[i],
                'label': label_id       # label 'number'
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path): #, frame_indices):
        clip = self.loader(path, trans=self.temporal_transform) #, frame_indices)
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

        #frame_indices = self.data[index]['frame_indices']
        
        # NO TEMPORAL TRANSFORMATION YET
        #if self.temporal_transform is not None:
        #    frame_indices = self.temporal_transform(frame_indices)
       
        
        clip = self.__loading(path) #, frame_indices)
  
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)

#------------------------------------------------------------------------
import json
import copy
import functools

import torch
from torch.utils.data.dataloader import default_collate

from .videodataset import VideoDataset


def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]
    import numpy as np
    target_element = batch_targets[0]

    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets


class Kinetics700MultiClips(Kinetics700):

    def __loading(self, path, video_frame_indices):
        clips = []
        #segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
            #segments.append(
            #    [min(clip_frame_indices),
            #     max(clip_frame_indices) + 1])

        return clips   #, segments

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
import json
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np

from .loader import VideoLoader


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

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

    return video_ids, video_paths, annotations

# --------- add label noise -------------
def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C
#------------------------------------------
##

class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label',
                 #---- additional parameters for LNL -------
                 meta_num=1000, corruption_type=None, corruption_prob=None,
                 is_meta=False):
        
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter,
            meta_num, corruption_type, corruption_prob, #additional
            is_meta)  # additional

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter, 
                       meta_num, corruption_type, corruption_prob, is_meta):
        with annotation_path.open('r') as f:
            data = json.load(f)

        # how about split the meta and train here
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter)
        if subset == 'training':
            if is_meta == True:
                video_ids = video_ids[:meta_num]
                video_paths = video_paths[:meta_num]
                annotations = annotations[:meta_num]
            else:            # load train data only
                video_ids = video_ids[meta_num:]
                video_paths = video_paths[meta_num:]
                annotations = annotations[meta_num:]

        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        # need to split meta and train first!
        # then corrupt the labels separately


        #----- label corruption process : 1. define matrix ----------
        C = None
        num_classes = len(idx_to_class)   # for label corruption
        if corruption_type == 'unif':
            C = uniform_mix_C(corruption_prob, num_classes)
            self.C = C
        elif corruption_type == 'flip':
            C = flip_labels_C(corruption_prob, num_classes)
        print(C)
        #   temporal noise?
        #-------------------------------------------------------------

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
                # don't need to execute label mapping when 'corruption_type_train' is None
                #if corruption_type is not None:                
                if C is not None:
                    label_id = np.random.choice(num_classes, p=C[label_id])
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            if not video_path.exists():
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            # add samples to dataset
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
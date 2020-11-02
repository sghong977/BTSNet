import json
from pathlib import Path

import torch
import torch.utils.data as data

from .loader import VideoLoader
# VideoDataset for Epic

def get_class_labels(noun_or_verb):
    if 'verb' in noun_or_verb:
        class_num = 125
    elif 'noun' in noun_or_verb:
        class_num = 352
    else:
        class_num = 352
    class_labels_map = {}
    index = 0
    for class_label in range(0, class_num):
        class_labels_map[str(class_label)] = index
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

import numpy as np

class EpicKitchen(data.Dataset):

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
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter)
        noun_or_verb = str(annotation_path).split('/')[-1]
        class_to_idx = get_class_labels(noun_or_verb)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[str(label)] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = int(annotations[i]['label'])
                label_id = int(annotations[i]['label'])#class_to_idx[label]
            else:
                label = 'test'
                label_id = -1
            # idx_to_class[str(annotations[i]['label'])] = str(annotations[i]['label'])
            video_path = video_paths[i]
            if not video_path.exists():
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
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
    # print("-------------------")
    # print(len(batch_clips))
    # print(np.shape(batch_clips[0]))
    # # print(type(batch_clips))
    # print("-------------------")
    # print(len(batch_targets))
    # print(np.shape(batch_targets[0]))
    # batch_clips2 = []
    # batch_targets2 = []
    # for i in range(len(batch_clips)):
    #     if batch_clips[i].size(1) == 16:
    #         batch_clips2.append(batch_clips[i])
    #         batch_targets2.append(batch_targets[i])

        # print(np.shape(batch_clips[i]))
    # return torch.stack(batch_clips), torch.Tensor(batch_targets)
    # print(len(batch_clips2), "--------------")
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets


class EpicKitchenMultiClips(EpicKitchen):

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
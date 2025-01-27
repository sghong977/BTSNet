from torchvision import get_image_backend
import torchvision

from datasets.videodataset import VideoDataset
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from datasets.activitynet import ActivityNet
from datasets.hollywood2 import Hollywood2, Hollywood2MultiClips
from datasets.charades import Charades, CharadesMultiClips
from datasets.epic_kitchen import EpicKitchen, EpicKitchenMultiClips
from datasets.loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5
from datasets.kinetics import Kinetics700, Kinetics700MultiClips



import torch

# --- class for concat multiple datasets ---------------
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
#--------------------------------------------------------

def image_name_formatter(x):
    return f'image_{x:05d}.jpg'

def jester_img_name_formatter(x):
    return f'{x:05d}.jpg'

# epic
def epic_image_name_formatter(x):
    return f'frame{x:010d}.jpg'
def epic_flow_name_formatter(flow, x):
    return flow + f'_{x:010d}.jpg'


def get_training_data(video_path,
                      annotation_path,
                      dataset_name,
                      input_type,
                      file_type,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None,
                      ):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 'jester', 'charades', 'SVW', 'hollywood2', 'epic'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / f'{video_id}.hdf5')

    if dataset_name == 'activitynet':
        training_data = ActivityNet(video_path,
                                    annotation_path,
                                    'training',
                                    data_name=dataset_name,
                                    spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform,
                                    video_loader=loader,
                                    video_path_formatter=video_path_formatter)
    elif dataset_name == 'jester':
        # different loader
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(jester_img_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(jester_img_name_formatter)
        video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)  #
        
        training_data = VideoDataset(video_path,
                                     annotation_path,
                                     'training',
                                     data_name=dataset_name,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)
    # different path (w/o label folder)
    elif dataset_name == 'hollywood2':
        video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)  #
        training_data = Hollywood2(video_path,
                                     annotation_path,
                                     'training',
                                     data_name=dataset_name,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)
    elif dataset_name == 'charades':
        video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)  #
        training_data = Charades(video_path,
                                     annotation_path,
                                     'training',
                                     data_name=dataset_name,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)
    elif dataset_name == 'epic':
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(epic_image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(epic_image_name_formatter)
        training_data = EpicKitchen(video_path,
                                     annotation_path,
                                     'training',
                                     data_name=dataset_name,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)
    elif dataset_name == 'kinetics':
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(epic_image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(epic_image_name_formatter)
        
        training_data = Kinetics700(video_path,
                                     annotation_path,
                                     'train',
                                     data_name=dataset_name,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)
    else:
        training_data = VideoDataset(video_path,
                                     annotation_path,
                                     'training',
                                     data_name=dataset_name,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)

    return training_data


def get_validation_data(video_path,
                        annotation_path,
                        dataset_name,
                        input_type,
                        file_type,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 'jester', 'charades', 'SVW', 'hollywood2', 'epic'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']
    from datasets.videodataset_multiclips import collate_fn
    collate_fn = collate_fn

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / f'{video_id}.hdf5')

    if dataset_name == 'activitynet':
        validation_data = ActivityNet(video_path,
                                      annotation_path,
                                      'validation',
                                      data_name=dataset_name,
                                      spatial_transform=spatial_transform,
                                      temporal_transform=temporal_transform,
                                      target_transform=target_transform,
                                      video_loader=loader,
                                      video_path_formatter=video_path_formatter)
    elif dataset_name == 'jester':
        # different loader
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(jester_img_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(jester_img_name_formatter)
        video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)  #

        validation_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            'validation',
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)        
    # different path : w/o label
    elif dataset_name == 'hollywood2':
        from datasets.hollywood2 import collate_fn_val
        collate_fn = collate_fn_val

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)  #

        validation_data = Hollywood2MultiClips(
            video_path,
            annotation_path,
            'validation',
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)        
    elif dataset_name == 'charades':
        from datasets.charades import collate_fn_val
        collate_fn = collate_fn_val

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)  #

        validation_data = CharadesMultiClips(
            video_path,
            annotation_path,
            'validation',
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)        
    elif dataset_name == 'epic':
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(epic_image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(epic_image_name_formatter)

        validation_data = EpicKitchenMultiClips(
            video_path,
            annotation_path,
            'validation',
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)        
    elif dataset_name == 'kinetics':
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(epic_image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(epic_image_name_formatter)

        validation_data = Kinetics700(
            video_path,
            annotation_path,
            'validation',
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)        
    else:
        validation_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            'validation',
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)

    return validation_data, collate_fn

# different scenario when charades or hollywood2
def get_inference_data(video_path,
                       annotation_path,
                       dataset_name,
                       input_type,
                       file_type,
                       inference_subset,
                       spatial_transform=None,
                       temporal_transform=None,
                       target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 'charades', 'SVW', 'hollywood2', 'epic'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']
    assert inference_subset in ['train', 'val', 'test']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / f'{video_id}.hdf5')

    if inference_subset == 'train':
        subset = 'training'
    elif inference_subset == 'val':
        subset = 'validation'
    elif inference_subset == 'test':
        subset = 'inference'

    if dataset_name == 'activitynet':
        inference_data = ActivityNet(video_path,
                                     annotation_path,
                                     subset,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter,
                                     is_untrimmed_setting=True)
    elif dataset_name == 'hollywood2':
        from datasets.hollywood2 import collate_fn
        video_path_formatter = (
            lambda root_path, video_id: root_path / video_id)  #
        inference_data = Hollywood2MultiClips(   #MultiClips
            video_path,
            annotation_path,
            subset,
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)
    elif dataset_name == 'charades':
        from datasets.charades import collate_fn
        loader = VideoLoader(image_name_formatter)
        video_path_formatter = (
            lambda root_path, video_id: root_path / video_id)  #
        inference_data = CharadesMultiClips(   #MultiClips
            video_path,
            annotation_path,
            subset,
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)
    elif dataset_name == 'epic':
        from datasets.epic_kitchen import collate_fn
#        loader = VideoLoader(image_name_formatter)
#        video_path_formatter = (
#            lambda root_path, video_id: root_path / video_id)  #
        inference_data = CharadesMultiClips(   #MultiClips
            video_path,
            annotation_path,
            subset,
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)
    elif dataset_name == 'kinetics':
        from datasets.epic_kitchen import collate_fn
#        loader = VideoLoader(image_name_formatter)
#        video_path_formatter = (
#            lambda root_path, video_id: root_path / video_id)  #
        inference_data = CharadesMultiClips(   #MultiClips
            video_path,
            annotation_path,
            subset,
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)
    else:
        inference_data = VideoDataset(   #MultiClips
            video_path,
            annotation_path,
            subset,
            data_name=dataset_name,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter,
            target_type=['label'])

    return inference_data, collate_fn
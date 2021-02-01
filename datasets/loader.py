import io

import h5py
from PIL import Image

import cv2
import torch


class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class ImageLoaderAccImage(object):

    def __call__(self, path):
        import accimage
        return accimage.Image(str(path))


class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            if image_path.exists():
                video.append(self.image_loader(image_path))
        return video


class VideoLoaderHDF5(object):

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:
            video_data = f['video']

            video = []
            for i in frame_indices:
                if i < len(video_data):
                    video.append(Image.open(io.BytesIO(video_data[i])))
                else:
                    return video

        return video


class VideoLoaderFlowHDF5(object):

    def __init__(self):
        self.flows = ['u', 'v']

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:

            flow_data = []
            for flow in self.flows:
                flow_data.append(f[f'video_{flow}'])

            video = []
            for i in frame_indices:
                if i < len(flow_data[0]):
                    frame = [
                        Image.open(io.BytesIO(video_data[i]))
                        for video_data in flow_data
                    ]
                    frame.append(frame[-1])  # add dummy data into third channel
                    video.append(Image.merge('RGB', frame))

        return video

# input : video
class VideoCutLoader(object):
    #def __init__(self):
    #self.caps = [cv2.VideoCapture(str(video_path)) for video_path in self.video_paths]
    #self.images = [[capid, framenum] for capid, cap in enumerate(self.caps) for framenum in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]
    """
    def __getitem__(self, idx):
        capid, framenum = self.images[idx]
        cap = self.caps[capid]
        cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
        res, frame = cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        img_tensor = torch.from_numpy(img).permute(2,0,1).float() # /255, -mean, /std ... do your things with the image
        label_tensor = torch.as_tensor(label)
        return img_tensor, label_tensor
    """
    # trans = temporal_transform
    def __call__(self, video_path, trans=None):   #frame_indices
        cap = cv2.VideoCapture(str(video_path))
        clips = list()
        while True:
            res, frame = cap.read()
            if res == False:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            clips.append(img)
        
        frame_indices = list(range(1, len(clips)+1))
        if trans is not None:
            frame_indices = trans(frame_indices)
            c = list()
            for f in frame_indices:
                c.append(clips[f])
            clips = c
        #video_image_tensor = torch.from_numpy(img).permute(2,0,1).float() # /255, -mean, /std ... do your things with the image
        return clips

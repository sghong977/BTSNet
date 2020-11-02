import numpy as np
import json
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--verb_path',
                    default="/raid/video_data/epic/slowfast_frames/",
                    type=str,
                    help=('Path of video directory (jpg or hdf5).'
                            'Using to get n_frames of each video.'))
parser.add_argument('--noun_path',
                    default='noun',
                    type=str,
                    help=('noun or verb'))
parser.add_argument('--dst_path',
                    default='/raid/video_data/epic/slowfast_frames/annotation',
                    type=str,
                    help='Path of dst json file.')
parser.add_argument('--output_type',
                    default='s1',
                    type=str,
                    help='s1 or s2')

args = parser.parse_args()
f = open(os.path.join(args.noun_path, args.output_type + '.json'), 'r')
noun = json.load(f)
f.close()
f = open(os.path.join(args.verb_path, args.output_type + '.json'), 'r')
verb = json.load(f)
f.close()


print('verb uid ', len(verb['results'].keys()), 'noun uid ', len(noun['results'].keys()))

for uid in verb['results'].keys():

    verb['results'][uid]['noun'] = noun['results'][uid]['noun']    
submission_json = verb
output_name = 'seen.json'
if args.output_type == 's2':
    output_name = 'unseen.json'
f = open(os.path.join(args.dst_path, output_name), 'w')
json.dump(submission_json, f)
f.close()
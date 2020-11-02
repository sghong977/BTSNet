import os
import json
import numpy as np
import argparse
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return list(y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb',
                        default='',
                        type=str,
                        help=('rgb path (seen or unseen)'))
    parser.add_argument('--flow',
                        default='',
                        type=str,
                        help=('flow path (seen or unseen)'))
    parser.add_argument('--data_type',
                        default='seen',
                        type=str,
                        help=('seen or unseen'))
    parser.add_argument('--dst_path',
                        default='/raid/video_data/output/epic/submission',
                        type=str,
                        help=('save dst path'))
    args = parser.parse_args()
    
    f = open(os.path.join(args.rgb, args.data_type + '.json'))
    rgb_data = json.load(f)
    f.close()
    f = open(os.path.join(args.flow, args.data_type + '.json'))
    flow_data = json.load(f)
    f.close()

    # Softmax after Add!

    result_data = rgb_data
    for uid in rgb_data['results'].keys():
        verb_rgb_output = []
        noun_rgb_output = []
        for i in range(0, 125):
            verb_rgb_output.append(rgb_data['results'][uid]['verb'][str(i)])
        for i in range(0, 352):
            noun_rgb_output.append(rgb_data['results'][uid]['noun'][str(i)])
        verb_flow_output = []
        noun_flow_output = []
        for i in range(0, 125):
            verb_flow_output.append(flow_data['results'][uid]['verb'][str(i)])
        for i in range(0, 352):
            noun_flow_output.append(flow_data['results'][uid]['noun'][str(i)])

        verb_rgb_output = softmax(np.array(verb_rgb_output))
        noun_rgb_output = softmax(np.array(noun_rgb_output))
        verb_flow_output = softmax(np.array(verb_flow_output))
        noun_flow_output = softmax(np.array(noun_flow_output))
        for i in range(0, 125):
            result_data['results'][uid]['verb'][str(i)] = verb_rgb_output[i] + verb_flow_output[i] 
        for i in range(0, 352):
            result_data['results'][uid]['noun'][str(i)] = noun_rgb_output[i] + noun_flow_output[i] 

    f = open(os.path.join(args.dst_path, args.data_type + '.json'), 'w')
    json.dump(result_data, f)
    f.close()
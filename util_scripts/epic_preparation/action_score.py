import os
import csv
import json
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
parser = argparse.ArgumentParser()
parser.add_argument(
    '--noun_root_path', default='/video_data/outoput/epic/Conv3d', type=str,
    help='root path'
)
parser.add_argument(
    '--verb_root_path', default='/video_data/outoput/epic/Conv3d', type=str,
    help='root path'
)
parser.add_argument(
    '--annotation_root_path', default="/raid/video_data/epic/annotation/", type=str,
    help='annotation path'
)
parser.add_argument(
    '--epoch', default=200, type=int,
    help='trained model epoch'
)


class VideoRecord(object):
    def __init__(self, row):
        self._data = row
    @property
    def uid(self):
        return str(self._data[0]).split('/')[-1]
    @property
    def verb_label(self):
        return int(self._data[2])
    @property   
    def noun_label(self):
        return int(self._data[1])


def calculate_accuracy(noun_target, verb_target, label):
    noun = 0.0
    verb = 0.0
    action = 0.0
    noun_precision_recall_pred = []
    noun_precision_recall_gt = []
    verb_precision_recall_pred = []
    verb_precision_recall_gt = []
    action_precision_recall_pred = []
    action_precision_recall_gt = []
    for idx in range(1, len(label)):
        noun_class = []
       
        for i in range(0, len(noun_target['results'][label[idx].uid]['noun'])):
            noun_class.append(noun_target['results'][label[idx].uid]['noun'][str(i)])

        noun_precision_recall_pred.append(noun_class.index(max(noun_class)))
        noun_precision_recall_gt.append(label[idx].noun_label)

        if noun_class.index(max(noun_class)) == label[idx].noun_label:
            noun += 1


        verb_class = []
        for i in range(0, len(verb_target['results'][label[idx].uid]['verb'])):
            verb_class.append(verb_target['results'][label[idx].uid]['verb'][str(i)])

        verb_precision_recall_pred.append(verb_class.index(max(verb_class)))
        verb_precision_recall_gt.append(label[idx].verb_label)

        if verb_class.index(max(verb_class)) == label[idx].verb_label:
            verb += 1    

        action_precision_recall_pred.append(verb_class.index(max(verb_class)) + noun_class.index(max(noun_class)) + 352)
        action_precision_recall_gt.append(label[idx].verb_label + label[idx].noun_label + 352)
        if noun_class.index(max(noun_class)) == label[idx].noun_label and verb_class.index(max(verb_class)) == label[idx].verb_label:
            action += 1
    noun_precision_recall = list(precision_recall_fscore_support(np.array(noun_precision_recall_gt), np.array(noun_precision_recall_pred), average='macro'))
    verb_precision_recall = list(precision_recall_fscore_support(np.array(verb_precision_recall_gt), np.array(verb_precision_recall_pred), average='macro'))
    action_precision_recall = list(precision_recall_fscore_support(np.array(action_precision_recall_gt), np.array(action_precision_recall_pred), average='macro'))
    noun_precision_recall = [noun_precision_recall[i]*100 for i in range(2)]
    verb_precision_recall = [verb_precision_recall[i]*100 for i in range(2)]
    action_precision_recall = [action_precision_recall[i]*100 for i in range(2)]
    return verb / len(label) * 100, noun / len(label) * 100, action / len(label) * 100, verb_precision_recall * 100, noun_precision_recall * 100, action_precision_recall * 100
if __name__ == '__main__':
    args = parser.parse_args()

    noun_path = os.path.join(args.noun_root_path, 'val.json')
   
    verb_path = os.path.join(args.verb_root_path, 'val.json')
    
    f = open(noun_path, 'r')
    noun = json.load(f)
    f.close()
    f = open(verb_path, 'r')
    verb = json.load(f)
    f.close()

   
    f = open(os.path.join(args.annotation_root_path, 'val.csv'), 'r')
    reader = csv.reader(f)
    groundtruth = []
    for row in reader:
        groundtruth.append(VideoRecord(row))
    f.close()


    print("noun len : ", len(noun['results']))
    print("verb len : ", len(verb['results']))
    print("gt   len : ", len(groundtruth))

    accuracy_verb, accuracy_noun, accuracy_action, precision_recall_verb, precision_recall_noun, precision_recall_action = calculate_accuracy(noun, verb, groundtruth)
    
    print("verb {:.2f}% noun {:.2f}% action {:.2f} precision_verb {:.2f} precision_noun {:.2f} precision_action {:.2f} recall_verb {:.2f} recall_noun {:.2f} recall_action {:.2f}".format(accuracy_verb, accuracy_noun, accuracy_action, precision_recall_verb[0], precision_recall_noun[0], precision_recall_action[0], precision_recall_verb[1], precision_recall_noun[1], precision_recall_action[1]))
   
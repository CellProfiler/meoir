#!/usr/bin/env python

import sys
from optparse import OptionParser
import numpy as np
import cpp
from scipy.spatial.distance import cdist, cosine, euclidean, cityblock
from .profiles import Profiles
from .confusion import confusion_matrix, write_confusion
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

class NNClassifier(object):
    def __init__(self, features, labels, distance='cosine'):
        assert isinstance(labels, list)
        assert len(labels) == features.shape[0]
        self.features = features
        self.labels = labels
        self.distance = {'cosine': cosine, 'euclidean': euclidean,
                         'cityblock': cityblock}[distance]

    def classify(self, feature):
        all_zero = np.all(self.features == 0, 1)
        distances = np.array([self.distance(f, feature) if not z else np.inf
                              for f, z in zip(self.features, all_zero)])
        return self.labels[np.argmin(distances)]

# A second implementation, originally written to make it possible to
# incorporate SVA (now removed, but kept in the sva branch), but kept
# for now because it may be clearer than the implementation above.
def crossvalidate(profiles, true_group_name, holdout_group_name=None, 
                  train=NNClassifier, distance='cosine'):
    profiles.assert_not_isnan()
    keys = profiles.keys()
    true_labels = profiles.regroup(true_group_name)
    profiles.data = np.array([d for k, d in zip(keys, profiles.data) if tuple(k) in true_labels])
    profiles._keys = [k for k in keys if tuple(k) in true_labels]
    keys = profiles.keys()
    labels = list(set(true_labels.values()))

    if holdout_group_name:
        holdouts = profiles.regroup(holdout_group_name)
    else:
        holdouts = dict((k, k) for k in keys)
    
    true_indices = []
    pred_indices = []
    for ho in set(holdouts.values()):
        test_set_mask = np.array([tuple(holdouts[k]) == ho for k in keys], 
                                 dtype=bool)
        training_features = profiles.data[~test_set_mask, :]
        training_labels = [labels.index(true_labels[tuple(k)]) 
                           for k, m in zip(keys, ~test_set_mask) if m]

        model = train(training_features, training_labels, distance=distance)
        for k, f, m in zip(keys, profiles.data, test_set_mask):
            if not m:
                continue
            true = true_labels[k]
            predicted = labels[model.classify(f)]
            
            true_indices.append(labels.index(true))
            pred_indices.append(labels.index(predicted))
    
    true_indices = np.array(true_indices)
    pred_indices = np.array(pred_indices)

    nmi_score = normalized_mutual_info_score(true_indices, pred_indices)
    ami_score = adjusted_mutual_info_score(true_indices, pred_indices)
 
    return nmi_score, ami_score 
 

if __name__ == '__main__':
    parser = OptionParser("usage: %prog [-c] [-h HOLDOUT-GROUP] PROPERTIES-FILE PROFILES-FILENAME TRUE-GROUP")
    parser.add_option('-c', dest='csv', help='input as CSV', action='store_true')
    parser.add_option('-d', dest='distance', help='distance metric', default='cosine', action='store')
    parser.add_option('-H', dest='holdout_group', help='hold out all that map to the same holdout group', action='store')
    options, args = parser.parse_args()
    if len(args) != 3:
        parser.error('Incorrect number of arguments')
    properties_file, profiles_filename, true_group_name = args
    cpp.properties.LoadFile(properties_file)

    if options.csv:
       profiles = Profiles.load_csv(profiles_filename)
    else:
       profiles = Profiles.load(profiles_filename)

    nmi_score, ami_score = crossvalidate(profiles, true_group_name, options.holdout_group,
                              distance=options.distance)
    print 'ami score is %s' % ami_score
    print 'nmi score is %s' % nmi_score

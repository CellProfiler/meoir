#!/usr/bin/env python

import sys
from optparse import OptionParser
import numpy as np
import cpa
from scipy.spatial.distance import cdist, cosine, euclidean, cityblock
from .profiles import Profiles
from .confusion import confusion_matrix, write_confusion


class NNClassifier(object):
    def __init__(self, features, labels, distance='cosine'):
        assert isinstance(labels, list)
        assert len(labels) == features.shape[0]
        self.features = np.array(features)
        self.labels = labels
        
	self.classLabels = np.unique(self.labels)
	self.classLabels.sort()
        
	self.distance = {'cosine': cosine, 'euclidean': euclidean,
                         'cityblock': cityblock}[distance]

    def classify(self, feature):
        all_zero = np.all(self.features == 0, 1)
        distances = np.array([self.distance(f, feature) if not z else np.inf
                              for f, z in zip(self.features, all_zero)])
        return self.labels[np.argmin(distances)]

    def calcDistances(self, feature): 
        all_zero = np.all(self.features == 0, 1)
        distances = np.array([self.distance(f, feature) if not z else np.inf
                              for f, z in zip(self.features, all_zero)])
        
	classDistances = np.zeros(self.classLabels.shape)
        for classLabel in self.classLabels:
	    indexer = (self.labels == classLabel)
	    classDistances[classLabel] = np.min([self.distance(f, feature) if not z else np.inf
	                              for f, z in zip(self.features[indexer], all_zero[indexer])])
				       
	return classDistances

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

    confusion = {}
    treatment_confusion = {}
    crossentropy = 0
    accuracy = 0
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
	    trueIdx = labels.index(true)
	    distances = -1 * model.calcDistances(f)
	    tmp = np.exp(distances - np.max(distances))
	    probabilities = tmp / tmp.sum()
            crossentropy += -np.log(probabilities[trueIdx]) 
	    
	    accuracy += 1. if trueIdx == np.argmax(probabilities) else 0
	    for i, classLabel in enumerate(model.classLabels):
                treatment_confusion[k, labels[classLabel]] = treatment_confusion.get((k, labels[classLabel]), 0) \
		                                             + probabilities[i]
		confusion[true, labels[classLabel]] = confusion.get((true, classLabel), 0) + probabilities[i]
    for l1 in labels:
        norm = np.sum([confusion[l1, l2] for l2 in labels])
	for l2 in labels:
	    confusion[l1, l2] /= norm 
	    
    return confusion, treatment_confusion, crossentropy / profiles.data.shape[0], float(accuracy) / profiles.data.shape[0]

if __name__ == '__main__':
    parser = OptionParser("usage: %prog [-c] [-h HOLDOUT-GROUP] PROPERTIES-FILE PROFILES-FILENAME TRUE-GROUP")
    parser.add_option('-c', dest='csv', help='input as CSV', action='store_true')
    parser.add_option('-d', dest='distance', help='distance metric', default='cosine', action='store')
    parser.add_option('-H', dest='holdout_group', help='hold out all that map to the same holdout group', action='store')
    options, args = parser.parse_args()
    if len(args) != 3:
        parser.error('Incorrect number of arguments')
    properties_file, profiles_filename, true_group_name = args
    cpa.properties.LoadFile(properties_file)

    if options.csv:
       profiles = Profiles.load_csv(profiles_filename)
    else:
       profiles = Profiles.load(profiles_filename)

    confusion, treatment_confusion, crossentropy, accuracy = crossvalidate(profiles, true_group_name, options.holdout_group,
                              distance=options.distance)
    write_confusion(confusion, sys.stdout)
    print '#'
    write_confusion(treatment_confusion, sys.stdout)
    print '#'
    print 'cross entropy loss is %s' % crossentropy
    print 'accuracy is %s %%' % (accuracy * 100) 

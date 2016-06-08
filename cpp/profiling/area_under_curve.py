#!/usr/bin/env python

import sys
from optparse import OptionParser
import numpy as np
import cpa
from scipy.spatial.distance import cdist, cosine, euclidean, cityblock
from .profiles import Profiles
from .confusion import confusion_matrix, write_confusion
from scipy.misc import comb

class kNNClassifier(object):
    def __init__(self, features, labels, distance='cosine'):
        assert isinstance(labels, list)
	assert len(labels) == features.shape[0]
	self.features = features
	self.labels = labels
	self.distance = {'cosine': cosine, 'euclidean': euclidean,
	                 'cityblock': cityblock}[distance]

    def classify(self, feature, k = 1):
        assert len(self.labels) >= k
        all_zero = np.all(self.features == 0, 1)
        distances = np.array([self.distance(f, feature) if not z else np.inf
                              for f, z in zip(self.features, all_zero)])
        indices = distances.argsort()[:k]
	return [self.labels[idx] for idx in indices]

# A second implementation, originally written to make it possible to
# incorporate SVA (now removed, but kept in the sva branch), but kept
# for now because it may be clearer than the implementation above.
def crossvalidate(profiles, true_group_name, holdout_group_name=None, 
                  train=kNNClassifier, distance='cosine', k_range=xrange(1,6)):
    profiles.assert_not_isnan()
    keys = profiles.keys()
    true_labels = profiles.regroup(true_group_name)
    profiles.data = np.array([d for key, d in zip(keys, profiles.data) if tuple(key) in true_labels])
    profiles._keys = [key for key in keys if tuple(key) in true_labels]
    keys = profiles.keys()
    labels = list(set(true_labels.values()))
    

    if holdout_group_name:
        holdouts = profiles.regroup(holdout_group_name)
    else:
        holdouts = dict((key, key) for key in keys)

    random_accuracies = np.zeros((len(holdouts.values()), len(k_range)))
    knn_accuracies = np.zeros((len(holdouts.values()), len(k_range)))
    knn_accuracies2 = np.zeros(len(k_range)) 
    confusion = {}
    for i, ho in enumerate(set(holdouts.values())):
        test_set_mask = np.array([tuple(holdouts[key]) == ho for key in keys], 
                                 dtype=bool)
        training_features = profiles.data[~test_set_mask, :]
        training_labels = [labels.index(true_labels[tuple(key)]) 
                           for key, m in zip(keys, ~test_set_mask) if m]

        
	model = train(training_features, training_labels, distance=distance)
        for key, f, m in zip(keys, profiles.data, test_set_mask):
            if not m:
                continue
	    true = true_labels[key]
	    
	    num_correct = np.sum(np.array(training_labels) == labels.index(true))
	    exactComb = True
            for j, k in enumerate(k_range):
		predicted = [labels[idx] for idx in model.classify(f, k)]
		
		if true in predicted:
		    knn_accuracies[i, j] += 1.
		    knn_accuracies2[j] += 1.
                random_accuracies[i, j] += 1 - (comb(len(training_labels) - num_correct, k, exact = exactComb) \
	                             / float(comb(len(training_labels), k, exact = exactComb)))
		for predicted_label in predicted:
		    confusion[true, predicted_label] = confusion.get((true, predicted_label), 0) + 1
        
        random_accuracies[i, :] /= profiles.data.shape[0] - training_features.shape[0]
	knn_accuracies[i, :] /= profiles.data.shape[0] - training_features.shape[0]
    return confusion, random_accuracies[:i+1].mean(axis=0), knn_accuracies[:i+1].mean(axis=0)


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
    k_range = xrange(1,6)
    confusion, random_accuracies, knn_accuracies = crossvalidate(profiles, true_group_name, options.holdout_group,
                              distance=options.distance, k_range=k_range)
    write_confusion(confusion, sys.stdout)
    print '#'
    print '%5s | %6s | %6s' % ('k', 'rand', 'knn')
    print '-' * 25
    for (k, knn, rand) in zip(k_range, knn_accuracies, random_accuracies): 
	print '%5d | %4.1f %% | %4.1f %%' % (k, rand * 100, knn * 100)
    print 'mean difference is: %s' % np.mean(knn-rand)

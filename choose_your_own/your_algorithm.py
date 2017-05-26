#!/usr/bin/python
'''
foo bar
'''

from time import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

import matplotlib.pyplot as plt

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

# pylint: disable=invalid-name
features_train, labels_train, features_test, labels_test = makeTerrainData()


# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatterplot and identify them visually
grade_fast = [features_train[ii][0]
              for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1]
              for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0]
              for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1]
              for ii in range(0, len(features_train)) if labels_train[ii] == 1]


# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
##########################################################################

def evalutate(clf, show_picture=False, print_info=False):
    t0 = time()
    clf.fit(features_train, labels_train)
    if print_info:
        print "training time:", round(time()-t0,3), "s" 

    t0 = time()
    predition = clf.predict(features_test)
    if print_info:
        print "prediction time:", round(time()-t0,3), "s"

    accuracy = accuracy_score(labels_test, predition)
    if print_info:
        print "Accuracy is %.5f" % accuracy

    if show_picture:
        try:
            prettyPicture(clf, features_test, labels_test)
        except NameError:
            print "Couldn't show picture for some reason"

    return accuracy


# your code here!  name your classifier object clf if you want the
# visualization code (prettyPicture) to show you the decision boundary

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

best_accuracy_so_far = 0
best_classifier = None

used_weight = None
used_neighbors = 0

# K-nearest neighbors
print
print "KNeighborsClassifier..."
for wight in ['distance', 'uniform']:
    for neighbors in range(5,30,5):
        clf = KNeighborsClassifier(weights=wight, n_neighbors=neighbors)
        tmp_acc = evalutate(clf)
        if tmp_acc > best_accuracy_so_far:
            best_accuracy_so_far = tmp_acc
            best_classifier = clf
            used_weight = wight
            used_neighbors = neighbors

print "The winner used %d neighbors and %s for weights" % (used_neighbors, used_weight)
evalutate(best_classifier, True, True)

# random forest
print
print "RandomForestClassifier"
clf = RandomForestClassifier()
evalutate(clf, True, True)

# adaboost (sometimes also called boosted decision tree)
print
print "AdaBoostClassifier"
clf = AdaBoostClassifier()
evalutate(clf, True, True)



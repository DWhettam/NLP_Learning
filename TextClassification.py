import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode
import pickle

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
            self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = float(choice_votes)/len(votes)
        return mode(votes), conf


documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[:1900]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes accuracy:", nltk.classify.accuracy(classifier, testing_set) * 100, "%")

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy:", nltk.classify.accuracy(MNB_classifier, testing_set) * 100, "%")

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier accuracy:", nltk.classify.accuracy(BNB_classifier, testing_set) * 100, "%")

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_set)
print("LogisticRegression_Classifier accuracy:", nltk.classify.accuracy(LogisticRegression_Classifier, testing_set) * 100, "%")

SGD_Classifier = SklearnClassifier(SGDClassifier())
SGD_Classifier.train(training_set)
print("SGD_Classifier accuracy:", nltk.classify.accuracy(SGD_Classifier, testing_set) * 100, "%")

SVC_Classifier = SklearnClassifier(SVC())
SVC_Classifier.train(training_set)
print("SVC_Classifier accuracy:", nltk.classify.accuracy(SVC_Classifier, testing_set) * 100, "%")
#
# LinearSVC_Classifier = SklearnClassifier(LinearSVC())
# LinearSVC_Classifier.train(training_set)
# print("LinearSVC_Classifier accuracy:", nltk.classify.accuracy(LinearSVC_Classifier, testing_set) * 100, "%")

NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print("NuSVC_Classifier accuracy:", nltk.classify.accuracy(NuSVC_Classifier, testing_set) * 100, "%")



voted_classifier = VoteClassifier(classifier,
                                    MNB_classifier,
                                    BNB_classifier,
                                    LogisticRegression_Classifier,
                                    SGD_Classifier,
                                    SVC_Classifier,
                                    NuSVC_Classifier)

print("Voted classifer accuracy:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100, "%")
classification, confidence = voted_classifier.classify(testing_set[0][0])
print("Classification: ", classification, "%. Confidence: ", confidence*100, "%")
classification, confidence = voted_classifier.classify(testing_set[1][0])
print("Classification: ", classification, "%. Confidence: ", confidence*100, "%")
classification, confidence = voted_classifier.classify(testing_set[2][0])
print("Classification: ", classification, "%. Confidence: ", confidence*100, "%")
classification, confidence = voted_classifier.classify(testing_set[3][0])
print("Classification: ", classification, "%. Confidence: ", confidence*100, "%")
classification, confidence = voted_classifier.classify(testing_set[4][0])
print("Classification: ", classification, "%. Confidence: ", confidence*100, "%")

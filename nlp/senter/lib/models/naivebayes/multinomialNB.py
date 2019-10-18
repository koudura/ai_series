import math
import operator
from typing import List, Iterable, Tuple

from nlp.senter.lib.models.naivebayes.NB import NaiveBayes
from nlp.senter.lib.parsers.textparser import TextParser


class MultinomialNB(NaiveBayes):

    def __init__(self, dataset: Tuple[List, List], parser: TextParser, classes: Tuple = ('0', '1')):

        super().__init__(dataset, parser, classes)
        self.class_probabilities = {}

    def train(self, function="log"):
        super(MultinomialNB, self).train(function)

        assert len(self.X_train) == len(self.Y_train)
        assert len(self.classes) > 1

        label_size = len(self.Y_train)
        train_size = len(self.X_train)

        """
        Initialize class probabilities for each class.
        """
        self.class_probabilities = {x: 0 for x in self.classes}
        """
        Create empty bag of words(BoW) model for each class in naive-model.
        """
        self.model = {x: {} for x in self.classes}
        """
        Iterate through training data-set for each training-sentence to create BoW model.
        """
        for i in range(train_size):
            """
            For each label increment its corresponding count for class probability.
            """
            self.class_probabilities[self.Y_train[i]] += 1
            """
            fill BoW model.
            """
            for word in self.X_train[i]:
                if word in self.model[self.Y_train[i]]:
                    self.model[self.Y_train[i]][word] += 1
                else:
                    self.model[self.Y_train[i]][word] = 1
            """
            Divide count of words per class by number labels, 
            to give probabilities of classes.
            """
            for _class in self.class_probabilities:
                self.class_probabilities[_class] /= label_size


    def test(self):
        pass

    def predict(self, sample):
        pass

    def fit(self, train_x: List, labels_y: List, classes: Iterable = ('0', '1')):
        """
        Fit training data and labels into naive bayes model. \n
        :type labels_y: List \n
        :type train_x: List \n
        :param train_x: training data-set to be fitted into model.
        :param labels_y: corresponding labels of training set to be fit into naive model.
        :param classes:
        :return: Fits object model with given data-set.
        """
        assert len(train_x) == len(labels_y)
        assert len(list(classes)) > 1

        """
        Initialize class probabilities for each class.
        """
        self.class_probabilities = {x: 0 for x in classes}
        """
        Create empty bag of words(BoW) model for each class in naive-model. 
        """
        self.model = {x: {} for x in classes}
        """
        Iterate through training data-set for each training-sentence to create BoW model.
        """
        for i in range(len(train_x)):
            """
            For each label increment its corresponding count for class probability. 
            """
            self.class_probabilities[labels_y[i]] += 1
            """
            fill BoW model.
            """
            for word in train_x[i]:
                if word in self.model[labels_y[i]]:
                    self.model[labels_y[i]][word] += 1
                else:
                    self.model[labels_y[i]][word] = 1
            """
            Divide count of words per class by number labels, 
            to give probabilities of classes.
            """
            for _class in self.class_probabilities:
                self.class_probabilities[_class] /= len(labels_y)

    def classify(self, sentence):
        """

        :param sentence:
        :return:
        """

        """

        """
        predict_model = {_class: {} for _class in self.class_probabilities}
        """

        """
        prediction = {_class: 1 for _class in self.class_probabilities}

        """

        """
        for word in sentence:
            for _class in self.model:
                if word in self.model[_class]:
                    predict_model[_class][word] = self.model[_class][word] + 1
                else:
                    predict_model[_class][word] = 1

        """
        For each class(c) in classes in naive model evaluate prediction
        via naive-bayes model-> || P(c|X) = (P(X|c) * P(c) / P(X) || 
        """
        for _class in predict_model:
            for word in predict_model[_class]: ...

import math
import operator
from typing import List, Tuple

from nlp.senter.lib.models.naivebayes.NB import NaiveBayes
from nlp.senter.lib.parsers.textparser import TextParser


class MultinomialNB(NaiveBayes):

    def __init__(self, dataset: Tuple[List, List], parser: TextParser, classes: Tuple = ('0', '1')):

        super().__init__(dataset, parser, classes)

        self.class_probabilities = {}
        self.function = ''

    def train(self, function="log"):
        super(MultinomialNB, self).train(function)

        assert len(self.X_train) == len(self.Y_train)
        assert len(self.classes) > 1
        assert self.function in {"log", "laplace"}

        label_size = len(self.Y_train)
        train_size = len(self.X_train)

        self.function = function

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

    def predict(self, sample):
        super(MultinomialNB, self).predict(sample)

        return self.__log__(sample) if self.function == "log" else self.__laplace__(sample)

    def test(self):
        return [self.__log__(data) if self.function == "log" else self.__laplace__(data) for data in self.X_test]

    def __log__(self, sample: str) -> str:
        """
        Use the logarithmic probability scaling
        function to estimate probability of class predictors \n
        :type sample: str
        :param sample: use model to predict class of sample data.
        :return: predicted class value.
        """
        """
        initialize class predictor model : P(c|x)
        """
        predictor = {_class: self.class_probabilities[_class] for _class in self.class_probabilities}

        """
        estimate class prior probabilities given a sample data.
        """
        for word in sample:
            for c in self.model:
                if word in self.model[c]:
                    predictor[c] += math.log(self.model[c][word])
        """
        :return maximum predicted class probability. i.e.
        given classes = ['0', '1'], if P(0|sample) > P(1|sample) return '0', otherwise '1'.
        """
        return max(predictor.items(), key=operator.itemgetter(1))[0]

    def __laplace__(self, sample):
        """
        Using the laplacian function for bag of words and probability smoothening
        to estimate probability of class predictors \n
        :param sample:  use model to predict class of sample data.
        :return: predicted class value.
        """
        aprior_model = {x: {} for x in self.class_probabilities}
        predictor = {_class: self.class_probabilities[_class] for _class in self.class_probabilities}

        for word in sample:
            for _class in self.model:
                if word in self.model[_class]:
                    aprior_model[_class][word] = self.model[_class][word] + 1
                else:
                    aprior_model[_class][word] = 1

        """
        For each class(c) in classes in naive model evaluate prediction
        via naive-bayes model-> || P(c|X) = (P(X|c) * P(c) / P(X) || 
        """
        for c in aprior_model:
            for word in aprior_model[c]:
                if len(aprior_model) > 0:
                    aprior_model[c][word] /= (sum(self.model[c].values()) + len(self.model[c]) + 1)
                    """
                    || P(x|c) || -> :A
                    """
                    predictor[c] *= aprior_model[c][word]
            """
            || P(x|c) * P(c) || :A * P(c) -> :B
            """
            predictor[c] *= self.class_probabilities[c]
            """
            || P(x|c) * P(c) / P(x) || :B / P(x) -> :C
            """
            if len(aprior_model[c] > 0):
                predictor[c] /= sum(aprior_model[c].values())

        """
        :return maximum predicted class probability. i.e.
        given classes = ['0', '1'], if P(0|sample) > P(1|sample) return '0', otherwise '1'.
        """
        return max(predictor.items(), key=operator.itemgetter(1))[0]

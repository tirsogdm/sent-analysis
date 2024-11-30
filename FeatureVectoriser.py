# PYTHON MODULES
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline
from abc import ABC, abstractmethod
from sklearn import metrics


class FeatureVectoriser(ABC):
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def run(self):
        cv = CountVectorizer()
        train_counts = cv.fit_transform(self.train_data)
        train_counts = self.vectorise(train_counts)

        clf = MultinomialNB().fit(train_counts, self.train_labels)
        test_counts = cv.transform(self.test_data)
        test_counts = self.vectorise(test_counts)
        predictions = clf.predict(test_counts)
        return predictions
    
    @abstractmethod
    def vectorise(self, data):
        """
        Receive output from CountVectorizer and apply any additional processing. Return counts.
        """
        pass

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self, train_set, eval_set):
        self.train_data, self.train_labels = train_set.flatten()
        self.eval_data, self.eval_labels = eval_set.flatten()

    def train(self):
        """
        Train the classifier.
        """
        train_counts = self.get_feature_counts(self.train_data, is_test=False)
        self.clf = MultinomialNB().fit(train_counts, self.train_labels)
        return train_counts

    def evaluate(self, verbose=True):
        """
        Evaluate the classifier.
        """
        eval_counts = self.get_feature_counts(self.eval_data, is_test=True)
        predictions = self.clf.predict(eval_counts)

        if verbose:
            print(metrics.confusion_matrix(self.eval_labels, predictions))
            print(metrics.classification_report(self.eval_labels, predictions))

        return predictions

    @abstractmethod
    def get_feature_counts(self, data, is_test=False):
        """
        Generate features from data. Customise in each instance to experiment with different feature counts.
        """
        if not is_test:
            self.cv = CountVectorizer()
            return self.cv.fit_transform(data)
        return self.cv.transform(data)

    # --- NORMALISATION METHODS ---
    @staticmethod
    def row_wise(counts):
        """
        Normalise counts by row.
        """
        return counts / counts.sum(axis=1)

    @staticmethod
    def tfidf(counts):
        """
        Normalise counts by term frequency inverse document frequency.
        """
        return counts
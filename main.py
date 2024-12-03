from sklearn.feature_extraction.text import CountVectorizer

from spacy.tokens import Doc, Token

import pickle

from components.Classifier import Classifier
from components.Pipeline import Pipeline
from components.Dataset import Dataset
from NaiveBayes import NaiveBayes

from utils import read_in

# Setting filtered attribute on Token and processed_text attribute on Doc
def filtered(token, dataset):
    """
    Method called on each token to check if it should be filtered.
    """
    if token.is_punct or token.is_space:
        return True

def processed_text(doc, dataset):
    """
    Method called on each doc to return the processed text containing only non-filtered tokens.
    """
    text = []
    for token in doc:
        if not token._.filtered(dataset):
            text.append(token.text)
    return " ".join(text)

Token.set_extension('filtered', method=filtered)
Doc.set_extension('processed_text', method=processed_text)


# Custom classifer for sepcifying feature generation
class CustomClassifier(Classifier):
    def __init__(self, train_set, eval_set):
        super().__init__(train_set, eval_set)
    
    def get_feature_counts(self, data, is_test=False):
        """
        Simple approach returning count vectors for each review.
        """
        if not is_test:
            self.cv = CountVectorizer()
            return self.cv.fit_transform(data)
        return self.cv.transform(data)


if __name__ == "__main__":
    reviews = read_in('data/pos', 1)
    reviews += read_in('data/neg', -1)

    # Pipeline
    pipeline = Pipeline()
    dataset = Dataset([1, -1], data=reviews, pipeline=pipeline)
    train_set, eval_set, test_set = dataset.split()

    classifier = CustomClassifier(train_set, eval_set)
    train_counts = classifier.train()
    # print(train_counts.shape)
    # prediction = classifier.evaluate()

    naive_bayes = NaiveBayes()
    naive_bayes.train(train_counts, train_set.flatten()[1])

    eval_data, eval_labels = eval_set.flatten()
    eval_counts = classifier.get_feature_counts(eval_data, is_test=True)
    predictions = naive_bayes.predict(eval_set)
    print(predictions)

    # train_labels = train_set.flatten()[1]
    # with open('counts_and_labels.pkl', 'wb') as file:
    #   pickle.dump((train_counts, train_labels, eval_set, classifier.cv), file)
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.Classifier import Classifier

"""
Experiment 1
------------
- Words as features, no word normalisation
- No filtering, phrases, normalisation
"""
description = "Words as features, no word normalisation, no filtering, no phrases, no normalisation"

def filtered(token, dataset):
    """
    Method called on each token to check if it should be filtered.
    """
    return False

def processed_text(doc, dataset):
    """
    Method called on each doc to return the processed text containing only non-filtered tokens.
    """
    text = []
    for token in doc:
        if not token._.filtered(dataset):
            text.append(token.text)
    return " ".join(text)

# Custom classifer for sepcifying feature generation
class CustomClassifier(Classifier):
    def __init__(self, train_set, eval_set):
        super().__init__(train_set, eval_set)
    
    def get_feature_counts(self, data, is_test=False):
        """
        Simple approach returning word count vectors for each review.
        """
        if not is_test:
            self.cv = CountVectorizer()
            counts = self.cv.fit_transform(data)
            return counts
        
        counts = self.cv.transform(data)
        return counts
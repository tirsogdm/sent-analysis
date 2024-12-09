from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.Classifier import Classifier

"""
Experiment 15
------------
- Word lemmas as features
- Filtering with (10, 250) frequency range
- Incorporating unigrams, bigrams, trigrams as features
- Tf-idf normalisation
"""
description = "Lemmas as features, filtering with (5, 250) frequency range, incorporating unigrams, bigrams, trigrams, and tfidf normalisation"

def filtered(token, dataset):
    """
    Method called on each token to check if it should be filtered.
    """
    return not dataset.in_freq_range(token.lemma_, min_df=5, max_df=250)

def processed_text(doc, dataset):
    """
    Method called on each doc to return the processed text containing only non-filtered tokens.
    """
    text = []
    for token in doc:
        if not token._.filtered(dataset):
            text.append(token.lemma_)
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
            # Unigrams
            self.cv_unigrams = CountVectorizer()
            self.cv_ngrams = CountVectorizer(ngram_range=(2, 3), min_df=5, max_df=250)

            counts_unigrams = self.cv_unigrams.fit_transform(data)
            counts_ngrams = self.cv_ngrams.fit_transform(self.train_set.raw_data)

            combined_counts = hstack([counts_unigrams, counts_ngrams])
            normalised_counts = self.tfidf(combined_counts)
            return normalised_counts
        
        counts_unigrams = self.cv_unigrams.transform(data)
        counts_ngrams = self.cv_ngrams.transform(self.eval_set.raw_data)
        combined_counts = hstack([counts_unigrams, counts_ngrams])
        normalised_counts = self.tfidf(combined_counts)
        return normalised_counts
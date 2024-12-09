from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import numpy as np

from abc import ABC, abstractmethod

from .CustomNB import CustomNB

class Classifier(ABC):
    """
    Abstract class for training and evaluating a given feature set. Define feature set in abstract method.
    """
    def __init__(self, train_set, eval_set):
        self.train_set = train_set
        self.eval_set = eval_set
        
        self.train_data, self.train_labels = train_set.flatten()
        self.eval_data, self.eval_labels = eval_set.flatten()

    def run(self):
        """
        Run the classifier.
        """
        self.train()
        self.evaluate()

    def train(self, type='sklearn'):
        """
        Train the classifier.

        Returns
        -------
        train_counts : list
            Feature counts of the training data.
        """
        train_counts = self.get_feature_counts(self.train_data, is_test=False)
        print("Feature vector size:", train_counts.shape[1])
        self.clf = MultinomialNB() if type == 'sklearn' else CustomNB()
        self.clf.fit(train_counts, self.train_labels)
        return train_counts

    def evaluate(self, verbose=True):
        """
        Evaluate the classifier.

        Parameters
        ----------
        verbose : bool
            Print evaluation metrics if True.
        
        Returns
        -------
        predictions : list
            Predictions of the classifier.
        """
        eval_counts = self.get_feature_counts(self.eval_data, is_test=True)
        predictions = self.clf.predict(eval_counts)

        if verbose:
            print(metrics.confusion_matrix(self.eval_labels, predictions))
            print(metrics.classification_report(self.eval_labels, predictions))

        return predictions

    # --- FEATURE GENERATION AND EXTRACTION METHODS ---
    @abstractmethod
    def get_feature_counts(self, data, is_test=False):
        """
        Generate features from data. Customise in each instance to experiment with different feature counts and normalisation methods.
        """
        if not is_test:
            self.cv = CountVectorizer()
            return self.cv.fit_transform(data)
        return self.cv.transform(data)

    # --- NORMALISATION METHODS ---
    def row_wise_l1(self, counts):
        """
        Normalise counts row-wise - L1 norm (sum of absolute values)

        Parameters
        ----------
        counts : scipy.sparse.csr.csr_matrix
            Sparse matrix of feature counts.
        
        Returns
        -------
        normalised_counts : scipy.sparse.csr.csr_matrix
            Normalised sparse matrix of feature counts.
        """
        # Sum of absolute values to get L1 norm
        row_norms = counts.sum(axis=1)
        # Flatten to a 1D array
        row_norms = np.array(row_norms).flatten()
        # Avoid any division by zero
        row_norms[row_norms == 0] = 1
        # Reshape to column vector
        row_norms = row_norms[:, None]

        # Element-wise division of each element by its row norm
        normalised_counts = counts.multiply(1 / row_norms)
        return normalised_counts

    def row_wise_l2(self, counts):
        """
        Normalise counts row-wise - L2 norm (square root of sum of squares)

        Parameters
        ----------
        counts : scipy.sparse.csr.csr_matrix
            Sparse matrix of feature counts.
        
        Returns
        -------
        normalised_counts : scipy.sparse.csr.csr_matrix
            Normalised sparse matrix of feature counts.
        """
        # Square root of sum of squares to get L2 norm
        row_norms = np.sqrt(counts.multiply(counts).sum(axis=1))
        # Flatten to a 1D array
        row_norms = np.array(row_norms).flatten()
        # Avoid any division by zero
        row_norms[row_norms == 0] = 1
        # Reshape to column vector
        row_norms = row_norms[:, None]

        # Element-wise division of each element by its row norm
        normalised_counts = counts.multiply(1 / row_norms)
        return normalised_counts
 
    def tfidf(self, counts):
        """
        Normalise counts - Term Frequency Inverse Document Frequency.

        Parameters
        ----------
        counts : scipy.sparse.csr.csr_matrix
            Sparse matrix of feature counts.
        
        Returns
        -------
        tfidf : scipy.sparse.csr.csr_matrix
            Normalised feature counts.
        """
        # Term Frequency TF(d, t) = count of t in d / length of d
        # > Sum row-wise for document length (num of total term counts)
        doc_lengths = np.array(counts.sum(axis=1)).flatten()
        # > Element-wise division of each element by its row length
        tf = counts.multiply(1 / doc_lengths[:, None])

        # Inverse Document Frequency IDF(t) = log ( num of docs / DF(t) + epsilon ) where DF(t) = num docs in which t appears
        num_docs = counts.shape[0]
        # > Column-wise sum of non-zero counts
        doc_freq = np.array((counts >0).sum(axis=0)).flatten()
        # > Smooth values and avoid division by zero.
        idf = np.log((1 + num_docs) / (1 + doc_freq)) + 1

        # Term Frequency Inverse Document Frequency TFIDF(t,d) = TF(t,d) * IDF(t)
        tfidf = tf.multiply(idf)
        return tfidf
    

if __name__ == "__main__":
    clf = MultinomialNB()
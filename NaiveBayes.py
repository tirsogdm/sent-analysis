from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pickle

class NaiveBayes:
    def __init__(self):
        self.prior_probabilities = None
        self.feature_probabilities = None
        self.alpha = 1.0

    def train(self, counts, labels):
        """
        Train the model on the input counts and labels.

        Parameters
        ----------
        counts : list
            Feature counts of the training data.
        
        labels : list
            Labels of the training data.
        """
        print("----> reviews", counts.shape[0], "----> features", counts.shape[1])

        self.counts = counts
        self.labels = np.array(labels)
        self.n_samples = self.counts.shape[0]
        self.n_features = self.counts.shape[1]
        self.n_neg_samples = np.sum(self.labels == -1)
        self.n_pos_samples = np.sum(self.labels == 1)
        self.prior_probabilities = np.array([self.n_neg_samples, self.n_pos_samples]) / self.n_samples
        print(self.prior_probabilities)

        probs = self.set_feature_probabilities()
        return probs

    def set_feature_probabilities(self):
        """
        Compute feature probabilities for each class, i.e., p(feature | class).
        
        p(feature | class) = (count(feature, class) + 1) / (count(class) + 2)
        """
        class_labels = np.array([-1, 1])
        # Converting sparse matrix to csc - compressed sparse column - format for efficient column-wise operations
        compressed_counts = self.counts.tocsc()
        # Initialise feature count vectors for each class
        feature_counts = np.zeros((class_labels.shape[0], compressed_counts.shape[1]), dtype=float)
        # Iterate over each class label
        for label_idx, label in enumerate(class_labels):
            # Get indicies of reviews with current label
            label_indices = np.where(self.labels == label)[0]
            # Iterate over each feature index (column)
            for feature_idx in range(compressed_counts.shape[1]):
                # Extract single feature column - sparse vector (only non-zero values)
                feature_col = compressed_counts[:, feature_idx]
                # Match rows with index in sparse feature vector indices (i.e., non-zero value), and in label indices (i.e., belongs to current class)
                matching_non_zeros = np.intersect1d(feature_col.indices, label_indices)
                # Count number of matches
                feature_counts[label_idx, feature_idx] = len(matching_non_zeros)

        # Adjust features counts with Laplace smoothing
        feature_counts_smoothed = feature_counts + self.alpha

        # Adjust class counts with Laplace smoothing
        class_total_counts = np.array([self.n_neg_samples, self.n_pos_samples])
        class_total_counts_smoothed = class_total_counts + (self.alpha * self.n_features)
        
        # Compute probabilities
        self.feature_probabilities = feature_counts_smoothed / class_total_counts_smoothed[:, None]
        print("\nFeature probabilities (manual with Laplace smoothing):\n", self.feature_probabilities)
    

    def check_pfc(self, computed_proabilities):
        nb = MultinomialNB()
        nb.fit(self.counts.toarray(), self.labels)
        sklearn_log_probabilities = nb.feature_log_prob_
        sklearn_probabilities = np.exp(sklearn_log_probabilities)
        print("alpha", nb.alpha)
        print(sklearn_probabilities)

        # Ensure the shapes match
        assert computed_proabilities.shape == sklearn_probabilities.shape, \
            f"Shape mismatch: {computed_proabilities.shape} != {sklearn_probabilities.shape}"

        # Compute element-wise differences
        difference = np.abs(computed_proabilities - sklearn_probabilities)

        # Check if all differences are within an acceptable tolerance
        if np.allclose(computed_proabilities, sklearn_probabilities, atol=1e-6):
            print("Feature probabilities match scikit-learn!")
        else:
            print(f"Feature probabilities do not match. Max difference: {np.max(difference)}")

    # Helper functions
    # posterior probability
    def prev_predict_instance(self, x):
        """
        p(class | features) = log p(class) * prod(log p(f | class) for f in features)
        """
        log_probs = np.log(self.prior_probabilities)
        for class_idx in range(len(self.prior_probabilities)):
            log_probs[class_idx] *= np.sum(
                x * np.log(self.feature_probabilities[class_idx, :]) +
                (1 - x) * np.log(1 - self.feature_probabilities[class_idx, :])
            )
        return np.argmax(log_probs)

    def predict_instance(self, x):
        log_posterior = np.log(self.prior_probabilities)
        print(log_posterior)
        for class_idx in range(len(self.prior_probabilities)):
            # in log space multiplication becomes to addition
            present_features = x * np.log(self.feature_probabilities[class_idx, :])
            absent_features = (1 - x) * np.log(1 - self.feature_probabilities[class_idx, :])
            log_posterior[class_idx] += np.sum(present_features + absent_features)

        print(log_posterior)
        return np.argmax(log_posterior)

    def predict(self, X):
        return np.array([self.predict_instance(x) for x in X])


if __name__ == "__main__":    
    with open('counts_and_labels.pkl', 'rb') as file:
        train_counts, train_labels, eval_set, cv = pickle.load(file)

    naive_bayes = NaiveBayes()
    probs = naive_bayes.train(train_counts, train_labels)

    # print(train_counts[0])
    x = train_counts[0].toarray().flatten()
    predicted = naive_bayes.predict_instance(x)
    print(predicted, train_labels[0])
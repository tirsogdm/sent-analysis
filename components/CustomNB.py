import numpy as np

class CustomNB:
    def __init__(self):
        self.prior_probabilities = None
        self.feature_probabilities = None
        self.alpha = 1.0

    def fit(self, counts, labels):
        """
        Estimate feature probabilities from input counts and labels.

        Parameters
        ----------
        counts : scipy.sparse.csr.csr_matrix
            Sparse matrix of feature counts, size (n_samples, n_features).
        
        labels : list
            Labels for each instance in counts, size (n_samples,).
        """
        self.counts = counts
        self.labels = np.array(labels)
        self.class_labels = np.array([-1, 1])
        self.n_samples = self.counts.shape[0]
        self.n_features = self.counts.shape[1]
        self.n_neg_samples = np.sum(self.labels == -1)
        self.n_pos_samples = np.sum(self.labels == 1)

        # ESTIMATING PRIOR PROBABILITIES - p(class) = count(class) / count(all classes)
        self.prior_probabilities = np.array([self.n_neg_samples, self.n_pos_samples]) / self.n_samples

        # ESTIMATING FEATURE PROBABILITIES - p(feature | class) = (count(feature, class) + 1) / (count(class) + count(features))
        # > Converting sparse matrix to csc - compressed sparse column - format for efficient column-wise operations
        compressed_counts = self.counts.tocsc()
        # > Initialise feature count vectors for each class
        feature_counts = np.zeros((self.class_labels.shape[0], compressed_counts.shape[1]), dtype=float)
        # > Iterate over each class label
        for class_idx, label in enumerate(self.class_labels):
            # > Get indicies of reviews with current class label
            class_indices = np.where(self.labels == label)[0]
            # > Iterate over each feature index (column)
            for feature_idx in range(compressed_counts.shape[1]):
                # > Extract single feature column - sparse vector (only non-zero values)
                feature_col = compressed_counts[:, feature_idx]
                # > Match rows with index in sparse feature vector indices (i.e., non-zero value), and in class indices (i.e., belongs to current class)
                matching_non_zeros = np.intersect1d(feature_col.indices, class_indices)
                # > Count number of matches
                feature_counts[class_idx, feature_idx] = len(matching_non_zeros)

        # > Adjust features counts with Laplace smoothing (alpha = 1)
        feature_counts_smoothed = feature_counts + self.alpha

        # > Adjust class counts with Laplace smoothing
        class_total_counts = np.array([self.n_neg_samples, self.n_pos_samples])
        class_total_counts_smoothed = class_total_counts + (self.alpha * self.n_features)
        
        # > Compute probabilities
        self.feature_probabilities = feature_counts_smoothed / class_total_counts_smoothed[:, None]

    def predict_instance(self, features):
        """
        Predict class label for a single instance given input features f in x

        Parameters
        ----------
        features: scipy.sparse.csr.csr_matrix
            Sparse row feature vector, size (1, n_features)
        
        Returns
        -------
        int
            Predicted class label (-1 or 1)

        """
        # Initialise log posterior probabilities with log of prior probabilities
        log_posterior = np.log(self.prior_probabilities)

        # Iterate over each class
        for class_idx in range(len(self.prior_probabilities)):
            # > Extract probability of present (non-zero) features
            present_indices = features.indices # indices of non-zero features
            present_values = features.data # corresponding non-zero values
            # >> Proportional probability contributions of present features
            present_feature_probs = present_values * np.log(self.feature_probabilities[class_idx, present_indices])

            # > Extract probability of absent (zero valued) features
            absent_indices = np.setdiff1d(np.arange(self.n_features), present_indices) # indices of zero valued features
            # >> Inverse probability as contributions of absent features
            absent_feature_probs = np.log(1 - self.feature_probabilities[class_idx, absent_indices])

            # > Combine sum of log probabilities of present and absent features
            log_posterior[class_idx] += np.sum(present_feature_probs) + np.sum(absent_feature_probs)

        return self.class_labels[np.argmax(log_posterior)]

    def predict(self, counts):
        """
        Predict class label for each feature vector in input.

        Parameters
        ----------
        counts: scipy.sparse.csr.csr_matrix
            Sparse matrix of feature counts, size (n_samples, n_features).
        
        Returns
        -------
        np.array
            Predicted class labels for each instance.
        """
        return np.array([self.predict_instance(counts.getrow(i)) for i in range(counts.shape[0])])
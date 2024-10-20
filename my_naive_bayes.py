import numpy as np

class MyMultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Initialize the MyMultinomialNaiveBayes classifier.

        Parameters:
        - alpha (float): Smoothing parameter (Laplacian smoothing). Default is 1.0.
        """
        self.alpha = alpha  # Smoothing parameter
        self.class_priors = None  # Class prior probabilities
        self.feature_log_probs = None  # Log probabilities of features given class
        self.classes = None  # Unique class labels

    def fit(self, X, y):
        """
        Fit the MyMultinomialNaiveBayes model using training data.

        Parameters:
        - X (array-like): Feature matrix (term counts).
        - y (array-like): Target labels.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize priors and log probabilities
        self.class_priors = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_probs = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            # Calculate class prior: P(class) = count(class) / total_samples
            self.class_priors[idx] = X_c.shape[0] / n_samples
            # Calculate smoothed log probabilities for each feature
            smoothed_counts = X_c.sum(axis=0) + self.alpha
            total_count = smoothed_counts.sum()
            self.feature_log_probs[idx, :] = np.log(smoothed_counts / total_count)

    def predict(self, X):
        """
        Predict class labels for the given data.

        Parameters:
        - X (array-like): Feature matrix (term counts) to be predicted.

        Returns:
        - Array of predicted class labels.
        """
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        """
        Calculate posterior probabilities for each class and select the best class.

        Parameters:
        - x (array-like): A single sample (term counts).

        Returns:
        - Predicted class label.
        """
        posteriors = []

        for idx, c in enumerate(self.classes):
            # Calculate log(P(class)) + sum(log(P(feature | class) * count))
            log_prior = np.log(self.class_priors[idx])
            log_likelihood = np.sum(x * self.feature_log_probs[idx, :])
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
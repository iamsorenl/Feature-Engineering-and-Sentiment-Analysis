import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def binary_classification(df):
    """
    Convert the review score into binary classification labels.
    1: Positive (score >= 4), 0: Negative (score <= 2).
    Rows with score 3 are removed.
    """
    # Filter out reviews with score 3
    df = df[df['review/score'] != 3]
    # Create binary labels: 1 for positive, 0 for negative
    df['binary_review/score'] = df['review/score'].apply(lambda x: 1 if x >= 4 else 0)
    return df

def tt_split(df):
    """
    Split the dataset into training and testing sets.
    Use 85% of the data for training and 15% for testing.
    """
    # Use train_test_split to split the data by labels only
    y = df['binary_review/score']

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, 
        test_size=0.15,         # 15% for testing
        random_state=24,        # Ensure reproducibility
        shuffle=True,           # Shuffle data before splitting
        stratify=y              # Maintain class balance
    )
    
    return X_train, X_test, y_train, y_test

def plot_class_distribution(y_train, y_test):
    """
    Plot the distribution of labels in the train and test datasets.
    """
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train set distribution
    y_train.value_counts().plot(kind='bar', ax=axes[0], color='skyblue', title='Training Set Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')

    # Test set distribution
    y_test.value_counts().plot(kind='bar', ax=axes[1], color='lightgreen', title='Test Set Distribution')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')

    # Display the plots
    plt.show()

def naive_bayes(X_train, X_test, y_train, y_test):
    """
    Train a Naive Bayes classifier using 3 different feature engineering techniques
    on the training data and evaluate it on the test data.
    Features:
    1. Bag of Words (BoW)
    2. Term Frequency-Inverse Document Frequency (TF-IDF)
    3. N-grams
    """

    # Initialize the classifiers
    bow_classifier = MultinomialNB()
    tfidf_classifier = MultinomialNB()
    ngram_classifier = MultinomialNB()

    # Initialize the vectorizers
    bow_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))

    # Fit the vectorizers and transform the training data
    bow_train = bow_vectorizer.fit_transform(X_train['review/text'])
    tfidf_train = tfidf_vectorizer.fit_transform(X_train['review/text'])
    ngram_train = ngram_vectorizer.fit_transform(X_train['review/text'])

    # Train the classifiers
    bow_classifier.fit(bow_train, y_train)
    tfidf_classifier.fit(tfidf_train, y_train)
    ngram_classifier.fit(ngram_train, y_train)

    # Transform the test data
    bow_test = bow_vectorizer.transform(X_test['review/text'])
    tfidf_test = tfidf_vectorizer.transform(X_test['review/text'])
    ngram_test = ngram_vectorizer.transform(X_test['review/text'])

    # Predict the test data
    bow_pred = bow_classifier.predict(bow_test)
    tfidf_pred = tfidf_classifier.predict(tfidf_test)
    ngram_pred = ngram_classifier.predict(ngram_test)

    # Evaluate the classifiers
    bow_accuracy = accuracy_score(y_test, bow_pred)
    tfidf_accuracy = accuracy_score(y_test, tfidf_pred)
    ngram_accuracy = accuracy_score(y_test, ngram_pred)

    return bow_accuracy, tfidf_accuracy, ngram_accuracy
    

def main():
    # Create binary classification
    sbr = pd.read_csv('small_books_rating.csv')
    sbr_bc = binary_classification(sbr)
    print(sbr_bc.head())

    # Split the dataset
    X_train, X_test, y_train, y_test = tt_split(sbr_bc)
    
    # Print the class distribution
    print("\nTraining set label distribution:")
    print(y_train.value_counts())
    print("\nTest set label distribution:")
    print(y_test.value_counts())

    # Plot the class distribution
    plot_class_distribution(y_train, y_test)

    # Naive Bayes



if __name__ == "__main__":
    main()
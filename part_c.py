import pandas as pd
import numpy as np
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Import your custom Naive Bayes class
from my_naive_bayes import MyMultinomialNaiveBayes

def binary_classification(df):
    """Convert review scores into binary labels."""
    df = df[df['review/score'] != 3].copy()
    df['binary_review/score'] = df['review/score'].apply(lambda x: 1 if x >= 4 else 0)
    return df

def tt_split(df):
    """Split dataset into training and test sets."""
    y = df['binary_review/score']
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.15, random_state=24, shuffle=True, stratify=y
    )
    return X_train, X_test, y_train, y_test

def prepare_features(X_train, X_test):
    """Generate BoW, TF-IDF, and One-Hot Encoding features."""
    bow_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()
    enc = OneHotEncoder(handle_unknown='ignore')

    bow_train = bow_vectorizer.fit_transform(X_train['review/text'])
    bow_test = bow_vectorizer.transform(X_test['review/text'])

    tfidf_train = tfidf_vectorizer.fit_transform(X_train['review/text'])
    tfidf_test = tfidf_vectorizer.transform(X_test['review/text'])

    onehot_train = enc.fit_transform(X_train[['Title', 'review/summary']])
    onehot_test = enc.transform(X_test[['Title', 'review/summary']])

    return bow_train, bow_test, tfidf_train, tfidf_test, onehot_train, onehot_test

def evaluate_model(name, clf, X_train, X_test, y_train, y_test):
    """Train, predict, and evaluate the model."""
    # Measure training time
    start_train = timer()
    clf.fit(X_train, y_train)
    end_train = timer()

    # Measure prediction time
    start_pred = timer()
    y_pred = clf.predict(X_test)
    end_pred = timer()

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1_binary = f1_score(y_test, y_pred, average='binary')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"\n=== {name} ===")
    print(f"Training Time: {end_train - start_train:.4f} seconds")
    print(f"Prediction Time: {end_pred - start_pred:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Binary F1 Score: {f1_binary:.4f}")
    print(f"Macro-F1 Score: {f1_macro:.4f}")
    print("Confusion Matrix:")
    print(cm)

def apply_classifiers(classifiers, X_train, X_test, y_train, y_test):
    """Apply classifiers to the prepared data."""
    for name, clf, X_train_vec, X_test_vec in classifiers:
        evaluate_model(name, clf, X_train_vec, X_test_vec, y_train, y_test)

def main():
    # Load and preprocess data
    sbr = pd.read_csv('small_books_rating.csv')
    sbr_bc = binary_classification(sbr)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = tt_split(sbr_bc)

    # Prepare features
    bow_train, bow_test, tfidf_train, tfidf_test, onehot_train, onehot_test = prepare_features(X_train, X_test)

    # Define classifiers
    classifiers = [
        ("BoW + MyMultinomialNaiveBayes", MyMultinomialNaiveBayes(), bow_train, bow_test),
        ("TF-IDF + MyMultinomialNaiveBayes", MyMultinomialNaiveBayes(), tfidf_train, tfidf_test),
        ("One-Hot + MyMultinomialNaiveBayes", MyMultinomialNaiveBayes(), onehot_train, onehot_test)
    ]

    # Apply classifiers
    apply_classifiers(classifiers, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
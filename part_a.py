import pandas as pd
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def binary_classification(df):
    """
    Convert the review score into binary classification labels.
    1: Positive (score >= 4), 0: Negative (score <= 2).
    Rows with score 3 are removed.
    """
    # Filter out reviews with score 3
    df = df[df['review/score'] != 3].copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Create binary labels using .loc to modify safely
    df.loc[:, 'binary_review/score'] = df['review/score'].apply(lambda x: 1 if x >= 4 else 0)
    
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
    classifiers = [
        ("BoW", MultinomialNB(), CountVectorizer()),
        ("TF-IDF", MultinomialNB(), TfidfVectorizer()),
        ("N-Grams", MultinomialNB(), CountVectorizer(ngram_range=(1, 2))),
    ]

    for name, clf, vectorizer in classifiers:
        # Timer start
        start = timer()

        # Fit and transform the training data
        X_train_vec = vectorizer.fit_transform(X_train['review/text'])
        X_test_vec = vectorizer.transform(X_test['review/text'])

        # Train the model
        clf.fit(X_train_vec, y_train)

        # Prediction and timing end
        y_pred = clf.predict(X_test_vec)
        end = timer()

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        f1_binary = f1_score(y_test, y_pred, average='binary')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        # Print results
        print(f"\n=== {name} Model ===")
        print(f"Training + Prediction Time: {end - start:.4f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Binary F1 Score: {f1_binary:.4f}")
        print(f"Macro-F1 Score: {f1_macro:.4f}")
        print("Confusion Matrix:")
        print(cm)

def svm(X_train, X_test, y_train, y_test):
    """
    Train a Support Vector Machine (SVM) classifier using 3 different feature engineering techniques
    on the training data and evaluate it on the test data.
    Features:
    1. SVD + Linear Kernel
    2. Term Frequency-Inverse Document Frequency (TF-IDF) + Linear Kernel
    3. Term Frequency-Inverse Document Frequency (TF-IDF) + Polynomial Kernel
    """
    
    tfidf_vectorizer = TfidfVectorizer()
    svd = TruncatedSVD(n_components=100, random_state=24)

    # Transform the data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train['review/text'])
    tfidf_test = tfidf_vectorizer.transform(X_test['review/text'])
    svd_train = svd.fit_transform(tfidf_train)
    svd_test = svd.transform(tfidf_test)

    classifiers = [
        ("SVD + Linear Kernel", SVC(kernel='linear', random_state=24), svd_train, svd_test),
        ("TF-IDF + Linear Kernel", SVC(kernel='linear', random_state=24), tfidf_train, tfidf_test),
        ("TF-IDF + Polynomial Kernel", SVC(kernel='poly', degree=3, random_state=24), tfidf_train, tfidf_test),
    ]

    for name, clf, X_train_vec, X_test_vec in classifiers:
        # Timer start
        start = timer()

        # Train the model
        clf.fit(X_train_vec, y_train)

        # Prediction and timing end
        y_pred = clf.predict(X_test_vec)
        end = timer()

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        f1_binary = f1_score(y_test, y_pred, average='binary')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        # Print results
        print(f"\n=== {name} Model ===")
        print(f"Training + Prediction Time: {end - start:.4f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Binary F1 Score: {f1_binary:.4f}")
        print(f"Macro-F1 Score: {f1_macro:.4f}")
        print("Confusion Matrix:")
        print(cm)
    

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
    #plot_class_distribution(y_train, y_test)

    print("\n--start naive bayes--\n")

    # Train and evaluate the Naive Bayes classifier using different feature engineering techniques
    naive_bayes(X_train, X_test, y_train, y_test)

    print("\n--end naive bayes--\n")

    print("\n--start svm--\n")

    # Train and evaluate the SVM classifier using different feature engineering techniques
    svm(X_train, X_test, y_train, y_test)

    print("\n--end svm--\n")



if __name__ == "__main__":
    main()
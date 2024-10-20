from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def load_data_files(train_file, val_file, test_file_label):
    """
    Load the data from CSV files and return DataFrames.

    Parameters:
    train_file (str): Path to the training data CSV file.
    val_file (str): Path to the validation data CSV file.
    test_file_label (str): Path to the test data CSV file.

    Returns:
    tuple: A tuple containing three DataFrames (train_df, val_df, test_df).
    """
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file_label)
    return train_df, val_df, test_df

def create_vectorizer(ngram_range, stop_words, max_features, min_df, max_df):
    """
    Create and return a CountVectorizer with specified configuration.

    Parameters:
    ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams to be extracted.
    stop_words (str or list): If 'english', a built-in stop word list for English is used. If a list, it should contain the stop words.
    max_features (int): The maximum number of features to be extracted.
    min_df (int or float): When building the vocabulary, ignore terms that have a document frequency strictly lower than the given threshold.
    max_df (int or float): When building the vocabulary, ignore terms that have a document frequency strictly higher than the given threshold.

    Returns:
    CountVectorizer: Configured CountVectorizer instance.
    """
    vectorizer = CountVectorizer(
        ngram_range=ngram_range, 
        stop_words=stop_words, 
        max_features=max_features, 
        min_df=min_df, 
        max_df=max_df
    )
    return vectorizer

def vectorize_data(vectorizer, train_df, val_df, test_df):
    """
    Fit the vectorizer on training data and transform all datasets.

    Parameters:
    vectorizer (CountVectorizer): The CountVectorizer instance to be used for transforming the data.
    train_df (DataFrame): DataFrame containing the training data.
    val_df (DataFrame): DataFrame containing the validation data.
    test_df (DataFrame): DataFrame containing the test data.

    Returns:
    tuple: A tuple containing transformed data (X_train, X_val, X_test).
    """
    X_train = vectorizer.fit_transform(train_df['review'])
    X_val = vectorizer.transform(val_df['review'])
    X_test = vectorizer.transform(test_df['review'])
    return X_train, X_val, X_test

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Train the given model and evaluate it on the validation set.

    Parameters:
    model (sklearn.base.BaseEstimator): The model to be trained and evaluated.
    X_train (sparse matrix): Transformed training data.
    y_train (Series): Labels for the training data.
    X_val (sparse matrix): Transformed validation data.
    y_val (Series): Labels for the validation data.

    Returns:
    float: Validation accuracy of the model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{model.__class__.__name__} Validation Accuracy: {accuracy:.4f}")
    return accuracy

def evaluate_on_test(model, vectorizer, test_df, y_test, output_file):
    """
    Evaluate the best model on the labeled test set and save predictions to a CSV.

    Parameters:
    model (sklearn.base.BaseEstimator): The best model to be evaluated.
    vectorizer (CountVectorizer): The vectorizer used for transforming the data.
    test_df (DataFrame): DataFrame containing the test data.
    y_test (Series): Labels for the test data.
    output_file (str): Path to the output CSV file where predictions will be saved.

    Returns:
    float: Test accuracy of the model.
    """
    X_test = vectorizer.transform(test_df['review'])  # Use the best vectorizer
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} Test Accuracy: {accuracy:.4f}")

    # Save predictions alongside reviews
    predictions_df = test_df.copy()
    predictions_df['predicted_label'] = y_pred
    predictions_df[['review', 'label', 'predicted_label']].to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}")
    return accuracy

def main():
    # File paths
    train_file = 'train.csv'
    val_file = 'validation.csv'
    test_file = 'test.csv'

    # Load data
    train_df, val_df, test_df = load_data_files(train_file, val_file, test_file)
    y_train, y_val, y_test = train_df['label'], val_df['label'], test_df['label']

    # Optimal n-gram configurations for each model
    ngram_configs = [
        {'ngram_range': (1, 2), 'stop_words': 'english', 'max_features': 10000, 'min_df': 5, 'max_df': 0.75},  # LogisticRegression
        {'ngram_range': (1, 2), 'stop_words': 'english', 'max_features': 20000, 'min_df': 3, 'max_df': 0.8},   # SGDClassifier
        {'ngram_range': (1, 1), 'stop_words': None, 'max_features': 5000, 'min_df': 2, 'max_df': 0.7},         # RandomForestClassifier
        {'ngram_range': (1, 2), 'stop_words': 'english', 'max_features': 5000, 'min_df': 5, 'max_df': 0.9},    # MultinomialNB
        {'ngram_range': (1, 1), 'stop_words': 'english', 'max_features': 3000, 'min_df': 10, 'max_df': 0.6}    # DecisionTreeClassifier
    ]

    # Models
    models = [
        LogisticRegression(max_iter=1000, random_state=24),
        SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=24),
        RandomForestClassifier(n_estimators=100, random_state=24),
        MultinomialNB(),
        DecisionTreeClassifier(random_state=24)
    ]

    best_model = None
    best_accuracy = 0.0
    best_vectorizer = None  # Track the vectorizer of the best model

    # Train and evaluate each model with its own vectorizer
    for model, config in zip(models, ngram_configs):
        print(f"\nTraining {model.__class__.__name__} with n-gram configuration: {config}")

        # Create vectorizer for the current model
        vectorizer = create_vectorizer(**config)

        # Vectorize data using the current vectorizer
        X_train, X_val, _ = vectorize_data(vectorizer, train_df, val_df, test_df)

        # Train and evaluate the model
        val_accuracy = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_vectorizer = vectorizer  # Store the vectorizer used for the best model

    print(f"\nBest Model: {best_model.__class__.__name__} with Validation Accuracy: {best_accuracy:.4f}")

    # Evaluate the best model on the test set using its vectorizer
    evaluate_on_test(best_model, best_vectorizer, test_df, y_test, 'test_predictions.csv')

if __name__ == "__main__":
    main()
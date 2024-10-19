import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def load_data_files(train_file, val_file, test_file_label, test_file_no_label):
    """
    Load the data from CSV files and return DataFrames.
    
    Args:
        train_file (str): Path to the training CSV file.
        val_file (str): Path to the validation CSV file.
        test_file_label (str): Path to the labeled test CSV file.
        test_file_no_label (str): Path to the unlabeled test CSV file.
    
    Returns:
        tuple: A tuple containing four DataFrames (train, val, test_label, test_no_label).
    """
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_label_df = pd.read_csv(test_file_label)
    test_no_label_df = pd.read_csv(test_file_no_label)
    
    return train_df, val_df, test_label_df, test_no_label_df

def ngram_feature_extraction(train_df, val_df, test_label_df, test_no_label_df, ngram_range, stop_words, max_features, min_df, max_df):
    """
    Extract n-gram features from the text data using CountVectorizer.
    
    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_label_df (pd.DataFrame): Test data with labels.
        test_no_label_df (pd.DataFrame): Test data without labels.
        ngram_range (tuple): Range of n-grams to extract.
        stop_words (str): Stop words to filter out.
        max_features (int): Maximum number of features.
        min_df (int): Minimum document frequency.
        max_df (float): Maximum document frequency.
    
    Returns:
        tuple: A tuple containing four DataFrames (train, val, test_label, test_no_label).
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words, max_features=max_features, min_df=min_df, max_df=max_df)
    
    X_train = vectorizer.fit_transform(train_df['review'])
    X_val = vectorizer.transform(val_df['review'])
    X_test_label = vectorizer.transform(test_label_df['review'])
    X_test_no_label = vectorizer.transform(test_no_label_df['review'])
    
    return X_train, X_val, X_test_label, X_test_no_label

def main():
    train_file = 'train.csv'
    val_file = 'validation.csv'
    test_file_label = 'test_data_with_labels.csv'
    test_file_no_label = 'test_data_no_labels.csv'
    
    train_df, val_df, test_label_df, test_no_label_df = load_data_files(train_file, val_file, test_file_label, test_file_no_label)
    
    print("Data loaded successfully!")
    print("Training data:")
    print(train_df.head())
    print("\nValidation data:")
    print(val_df.head())
    print("\nTest data with labels:")
    print(test_label_df.head())
    print("\nTest data without labels:")
    print(test_no_label_df.head())

    # Extract n-gram features model 1
    ngram_range = (1, 1)
    stop_words = 'english'
    max_features = 5000
    min_df = 5
    max_df = 0.7
    Xtrain, Xval, Xtest_label, Xtest_no_label = ngram_feature_extraction(train_df, val_df, test_label_df, test_no_label_df, ngram_range, stop_words, max_features, min_df, max_df)

    # Extract n-gram features model 2
    ngram_range = (1, 2)
    stop_words = 'english'
    max_features = 10000
    min_df = 10
    max_df = 0.8
    Xtrain, Xval, Xtest_label, Xtest_no_label = ngram_feature_extraction(train_df, val_df, test_label_df, test_no_label_df, ngram_range, stop_words, max_features, min_df, max_df)
    # Extract n-gram features model 3
    ngram_range = (1, 3)
    stop_words = None
    max_features = 20000
    min_df = 20
    max_df = 0.9
    Xtrain, Xval, Xtest_label, Xtest_no_label = ngram_feature_extraction(train_df, val_df, test_label_df, test_no_label_df, ngram_range, stop_words, max_features, min_df, max_df)

    # Extract n-gram features model 4
    ngram_range = (1, 1)
    stop_words = None
    max_features = 10000
    min_df = 5
    max_df = 0.5
    Xtrain, Xval, Xtest_label, Xtest_no_label = ngram_feature_extraction(train_df, val_df, test_label_df, test_no_label_df, ngram_range, stop_words, max_features, min_df, max_df)

    # Extract n-gram features model 5
    ngram_range = (1, 2)
    stop_words = 'english'
    max_features = 5000
    min_df = 15
    max_df = 0.5
    Xtrain, Xval, Xtest_label, Xtest_no_label = ngram_feature_extraction(train_df, val_df, test_label_df, test_no_label_df, ngram_range, stop_words, max_features, min_df, max_df)



if __name__ == "__main__":
    main()
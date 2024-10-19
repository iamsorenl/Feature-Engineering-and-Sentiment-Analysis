import os
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_csvs_train_val(base_path, label_value):
    """
    Parses reviews from the pos and neg folders within train and assigns the corresponding label (1 for positive, 0 for negative).
    
    Args:
        base_path (str): Path to 'pos' or 'neg' folder.
        label_value (int): 1 for positive, 0 for negative.
    
    Returns:
        List of tuples containing (id, review, label).
    """
    reviews = []
    for filename in os.listdir(base_path):
        if filename.endswith(".txt"):
            with open(os.path.join(base_path, filename), 'r', encoding='utf-8') as f:
                review = f.read().strip()
                review_id = filename.split(".")[0]  # Extract the ID from the filename
                reviews.append((review_id, review, label_value))
    return reviews

# Parse reviews from 'train/pos' and 'train/neg'
train_pos = generate_csvs_train_val('aclImdb/train/pos', 1)
train_neg = generate_csvs_train_val('aclImdb/train/neg', 0)

# Combine the positive and negative reviews into one DataFrame
train_df = pd.DataFrame(train_pos + train_neg, columns=['id', 'review', 'label'])

# Split the data into 90% train and 10% validation
train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)

# Save the train and validation splits to CSV files
train_data.to_csv('train.csv', index=False)
val_data.to_csv('validation.csv', index=False)

print("Files 'train.csv' and 'validation.csv' created successfully!")
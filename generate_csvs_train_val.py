import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_reviews(base_path, label_value):
    """
    Parses reviews from a folder and assigns the corresponding label (1 for positive, 0 for negative).
    
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

# Step 1: Parse reviews from 'train/pos' and 'train/neg'
train_pos = parse_reviews('aclImdb/train/pos', 1)
train_neg = parse_reviews('aclImdb/train/neg', 0)

# Step 2: Combine positive and negative reviews into a DataFrame
df = pd.DataFrame(train_pos + train_neg, columns=['id', 'review', 'label'])

# Step 3: Shuffle the data to ensure randomness
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 4: Stratified split into 90% train and 10% validation sets
train_data, val_data = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

# Step 5: Save the splits to CSV files
train_data.to_csv('train.csv', index=False)
val_data.to_csv('validation.csv', index=False)

print("Balanced and stratified 'train.csv' and 'validation.csv' created successfully!")
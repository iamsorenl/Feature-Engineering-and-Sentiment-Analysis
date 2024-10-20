import os
import pandas as pd

def generate_csv_test(base_path, label_value):
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

# Parse reviews from 'test/pos' and 'test/neg'
test_pos = generate_csv_test('aclImdb/test/pos', 1)
test_neg = generate_csv_test('aclImdb/test/neg', 0)

# Combine the positive and negative reviews into one DataFrame
test_df = pd.DataFrame(test_pos + test_neg, columns=['id', 'review', 'label'])

# Shuffle the test data for randomness
test_df = test_df.sample(frac=1, random_state=24).reset_index(drop=True)

# Save the full test data with labels to CSV
test_df.to_csv('test.csv', index=False)

print("File 'test.csv' was created successfully!")
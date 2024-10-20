# Feature Engineering and Sentiment Analysis

**Naive Bayes, Support Vector Machine, and Decision Tree classifiers implemented using scikit-learn with n-gram feature extraction**

This project involves implementing and comparing multiple machine learning models on text data. The models include **Naive Bayes, SVM, and Decision Trees**, using feature extraction techniques such as **Bag-of-Words (BoW)**, **TF-IDF**, and **One-Hot Encoding**. Additionally, a custom Naive Bayes classifier is implemented to compare performance with scikit-learn's version.

---

## Project Structure

```plaintext
.
├── aclImdb/                     # Raw dataset containing IMDb movie reviews
│   ├── train/                   # Training data (positive/negative reviews)
│   ├── test/                    # Test data (positive/negative reviews)
├── hw1/                         # Homework solutions and Python scripts
│   ├── part_a.py                # Part A: Initial feature engineering & model evaluation
│   ├── part_b.py                # Part B: Sentiment analysis on Stanford dataset
│   ├── part_c.py                # Part C: Custom Naive Bayes implementation & comparison
│   ├── my_naive_bayes.py        # Custom Naive Bayes class implementation (Part C)
├── generate_csv_train.py        # Converts raw IMDb dataset into a train CSV
├── generate_csv_test.py         # Converts raw IMDb dataset into a test CSV
├── small_books_rating.csv       # Amazon book reviews dataset for Part A
├── test_predictions.csv         # Test predictions from the best-performing model
├── train.csv / test.csv         # Processed train and test CSVs for IMDb dataset
├── requirements.txt             # List of Python dependencies
└── README.md                    # Instructions and project details (this file)

Setup Instructions

1. Clone the Repository

Clone the project repository to your local machine:

git clone <repository-url>
cd <repository-folder>

2. Create a Virtual Environment

It’s recommended to use a virtual environment to manage dependencies:

# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

3. Install Dependencies

Install the required dependencies from the requirements.txt file:

pip install -r requirements.txt

Instructions for Running the Python Files

1. Part A: Feature Engineering & Initial Model Evaluation

File: part_a.py

Description:
This script loads the Amazon book reviews dataset (small_books_rating.csv), performs feature extraction using BoW, TF-IDF, and One-Hot Encoding, and evaluates Naive Bayes, SVM, and Decision Tree classifiers.
The models are compared based on accuracy, F1 scores, and confusion matrices.

Run:

python part_a.py

2. Part B: Sentiment Analysis on IMDb Dataset

File: part_b.py

Description:
This script performs sentiment classification on the IMDb movie review dataset. It uses n-gram features to train and evaluate models on a train/validation split. The script fine-tunes hyperparameters to find the optimal configurations and reports results for each model.

Run:

python part_b.py

3. Part C: Custom Naive Bayes Implementation

Files:

	•	part_c.py
	•	my_naive_bayes.py

Description:
This part implements a custom Naive Bayes classifier in my_naive_bayes.py. The part_c.py script integrates this custom classifier into the workflow to compare its performance with scikit-learn’s Multinomial Naive Bayes implementation.

Run:

python part_c.py

4. Generating Train/Test CSVs from Raw IMDb Dataset

Files:

	•	generate_csv_train.py
	•	generate_csv_test.py

Description:
These scripts process the raw IMDb dataset into structured train and test CSV files for use in other experiments.
Ensure the aclImdb folder is correctly placed when running these scripts.

Run:

# Generate train.csv
python generate_csv_train.py

# Generate test.csv
python generate_csv_test.py

Output Files

	•	test_predictions.csv: Contains predictions made on the IMDb test set by the best-performing model from Part B.
	•	train.csv / test.csv: Processed train and test splits generated from the IMDb dataset.

Summary of Results

Part A Results:

	•	Best Model: SVM with TF-IDF
	•	Accuracy: 89.02%
	•	Binary F1 Score: 0.8898

Part B Results:

	•	Best Model: Logistic Regression with bigrams
	•	Test Accuracy: 85.22%
	•	Validation Accuracy: 87.36%

Part C Results:

	•	Custom Naive Bayes Performance:
	•	BoW Accuracy: 85.02%
	•	TF-IDF Accuracy: 87.09%
	•	One-Hot Encoding Accuracy: 85.78%

Troubleshooting

	1.	Virtual Environment Activation Issues:
Ensure you are using the correct path to the virtual environment.
On Windows, try using venv\Scripts\activate.bat if the regular command doesn’t work.
	2.	Missing Dependencies:
Run pip install -r requirements.txt inside the virtual environment to install dependencies.
	3.	Dataset Issues:
Ensure the aclImdb dataset folder is in the correct directory when generating CSV files.

License

This project is licensed under the MIT License. See the LICENSE file for more information.

Acknowledgments

	•	The IMDb dataset is provided by Maas et al. (2011).
	•	Amazon book reviews dataset used for initial feature engineering tasks.
	•	Scikit-learn for pre-built models and utilities.
```

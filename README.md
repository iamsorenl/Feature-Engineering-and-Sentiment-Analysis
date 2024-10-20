# Feature Engineering and Sentiment Analysis

**Naive Bayes, Support Vector Machine, and Decision Tree Classifiers Implemented Using scikit-learn & Custom Models**

This project showcases **feature engineering techniques** and **sentiment analysis** using **Naive Bayes, SVM, and Decision Trees** on Amazon and IMDb datasets. It integrates **n-gram feature extraction** with models built using scikit-learn and includes a **custom Naive Bayes classifier** to compare performance with built-in models.

---

## Features

- **Custom Naive Bayes Implementation:** Developed from scratch and compared with scikit-learn’s version.
- **Multiple Feature Engineering Techniques:** BoW, TF-IDF, and One-Hot Encoding.
- **Sentiment Analysis on IMDb and Amazon Datasets.**
- **Performance Metrics:** Comparison of Accuracy, F1 Score, and Confusion Matrices for all models.

---

## Tech Stack

- [Python](https://www.python.org) - Programming language
- [scikit-learn](https://scikit-learn.org) - Machine learning library
- [pandas](https://pandas.pydata.org) - Data manipulation
- [Matplotlib](https://matplotlib.org) - Visualization library
- **IMDb & Amazon Datasets** - Datasets used for model evaluation

---

## Project Structure

```plaintext
.
├── aclImdb/                     # Raw IMDb dataset
│   ├── train/                   # Training data
│   ├── test/                    # Test data
├── hw1/                         # Python scripts for each part
│   ├── part_a.py                # Part A: Feature engineering & evaluation
│   ├── part_b.py                # Part B: Sentiment analysis on IMDb dataset
│   ├── part_c.py                # Part C: Custom Naive Bayes implementation
│   ├── my_naive_bayes.py        # Custom Naive Bayes classifier class
├── generate_csv_train.py        # Generate train CSV from IMDb data
├── generate_csv_test.py         # Generate test CSV from IMDb data
├── small_books_rating.csv       # Amazon book reviews dataset
├── train.csv / test.csv         # Processed IMDb datasets
├── test_predictions.csv         # Output predictions from models
├── requirements.txt             # List of dependencies
└── README.md                    # This file
```

---

## Installation

This project requires **Python 3.8+**. Follow the steps below to set up the environment and run the project.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Scripts

### Part A: Feature Engineering & Model Evaluation

**File:** `part_a.py`  
This script applies **BoW, TF-IDF, and One-Hot Encoding** on the Amazon book reviews dataset, evaluating the performance of **Naive Bayes, SVM, and Decision Tree models**.

```bash
python part_a.py
```

---

### Part B: Sentiment Analysis on IMDb Dataset

**File:** `part_b.py`  
This script analyzes sentiment on the **IMDb dataset** using n-gram features and multiple classifiers. It evaluates models with both a **train/validation split** and reports accuracy, F1 scores, and confusion matrices.

```bash
python part_b.py
```

---

### Part C: Custom Naive Bayes Implementation & Comparison

**Files:**

- `part_c.py`
- `my_naive_bayes.py`

This section demonstrates a **custom Naive Bayes classifier** and compares its performance against the scikit-learn version. The script in `part_c.py` integrates the custom model into the workflow.

```bash
python part_c.py
```

---

### Generating CSV Files from IMDb Data

**Files:**

- `generate_csv_train.py`
- `generate_csv_test.py`

These scripts convert the raw IMDb dataset into structured **CSV files** for further analysis. Ensure the `aclImdb` folder is placed correctly.

```bash
# Generate train.csv
python generate_csv_train.py

# Generate test.csv
python generate_csv_test.py
```

---

## Results Summary

| **Model**   | **Feature**      | **Accuracy** | **Binary F1 Score** |
| ----------- | ---------------- | ------------ | ------------------- |
| Naive Bayes | BoW              | 85.02%       | 0.8410              |
| Naive Bayes | TF-IDF           | 87.09%       | 0.8670              |
| Naive Bayes | One-Hot Encoding | 85.78%       | 0.8598              |

- **Best Model (Part A):** SVM with TF-IDF, achieving **89.02% accuracy**.
- **Best Model (Part B):** Logistic Regression with bigrams, **87.36% validation accuracy** and **85.22% test accuracy**.
- **Custom Naive Bayes (Part C):** Matches or outperforms scikit-learn Naive Bayes in multiple cases.

---

## Troubleshooting

1. **Virtual Environment Issues:**  
   Ensure the virtual environment is properly activated.

   - On Windows: Use `venv\Scripts\activate.bat` if necessary.

2. **Dataset Issues:**  
   Ensure the `aclImdb` folder is placed correctly before generating CSVs.

3. **Dependency Issues:**  
   Install dependencies again if there are missing packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Contributing

Want to contribute? Great!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## Acknowledgments

- **IMDb Dataset:** Provided by Maas et al. (2011).
- **Amazon Book Reviews Dataset:** Used for feature engineering.
- **scikit-learn:** For machine learning utilities.

---

## Contact

For questions or suggestions, contact:  
**Soren Larsen** – [iamsorenl@gmail.com](mailto:iamsorenl@gmail.com)

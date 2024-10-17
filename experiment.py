import pandas as pd

sbr = pd.read_csv('small_books_rating.csv')

# Display the first few rows of the dataframe
print("First few rows of the dataframe:")
print(sbr.head())

# Open up review/score from the csv
review_text = sbr['review/text']
review_score = sbr['review/score']
review_titles = sbr['Title']
review_summary = sbr['review/summary']

print("\nFirst few rows of review_text:")
print(review_text.head())
print("\nfirst few rows of review_score:")
print(review_score.head())
print("\nTitles of the books:")
print(review_titles.head())
print("\nSummary of the books:")
print(review_summary.head())
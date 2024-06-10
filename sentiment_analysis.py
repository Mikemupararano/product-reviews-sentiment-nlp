import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Add the TextBlob sentiment analysis pipeline component to spaCy
nlp.add_pipe("spacytextblob")

# Load the dataset
dataframe = pd.read_csv('amazon_product_reviews.csv')

# Preprocess the text data: select the 'review.text' column and remove missing values
reviews_data = dataframe['reviews.text']
clean_data = reviews_data.dropna()

# Function for text cleaning
def clean_text(text):
    doc = nlp(text)
    tokens = [token.text.lower().strip() for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Apply text cleaning to the reviews
clean_data = clean_data.apply(clean_text)

# Function for sentiment analysis
def analyze_sentiment(review):
    doc = nlp(review)
    polarity = doc._.blob.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Test the sentiment analysis function on a few sample reviews
sample_reviews = clean_data.sample(5)
for review in sample_reviews:
    sentiment = analyze_sentiment(review)
    print(f"Review: {review}\nSentiment: {sentiment}\n")

# Save sample results to a file for verification
with open('sample_sentiment_results.txt', 'w') as f:
    for review in sample_reviews:
        sentiment = analyze_sentiment(review)
        f.write(f"Review: {review}\nSentiment: {sentiment}\n\n")

# Compare the similarity of two reviews
review1 = nlp(clean_data.iloc[0])
review2 = nlp(clean_data.iloc[1])
similarity_score = review1.similarity(review2)
print(f"Similarity between review 1 and review 2: {similarity_score}")

# Save similarity results to a file
with open('review_similarity_results.txt', 'w') as f:
    f.write(f"Review 1: {clean_data.iloc[0]}\n")
    f.write(f"Review 2: {clean_data.iloc[1]}\n")
    f.write(f"Similarity score: {similarity_score}\n")

# Sentiment Analysis Report

## Dataset Description
The dataset used for this analysis is the `Consumer Reviews of Amazon Products`. It contains customer reviews of various products sold on Amazon. The primary column of interest for this analysis is `review.text`, which contains the text of the product reviews.

## Preprocessing Steps
1. **Loading Data**: The dataset was loaded using Pandas, focusing on the `review.text` column.
2. **Cleaning Data**: Missing values in the `review.text` column were removed.
3. **Text Cleaning**: Reviews were processed to remove stopwords and non-alphabetical characters. The text was also converted to lowercase and stripped of leading/trailing whitespace.

## Evaluation of Results
The sentiment analysis was tested on a few sample product reviews, and the results were saved in a file for verification. The polarity scores determined whether the sentiment was positive, negative, or neutral. The analysis also included comparing the similarity of two sample reviews using the spaCy similarity function.

## Insights
The model effectively identified positive and negative sentiments in product reviews. However, the accuracy of sentiment detection can be affected by the complexity of the text and the presence of sarcasm or idiomatic expressions. The similarity function provided useful insights into how closely related two reviews were, which can be helpful for clustering similar reviews together.

## Model Strengths and Limitations
### Strengths
- Easy to implement and integrate with existing spaCy pipelines.
- Efficient text preprocessing and sentiment analysis.

### Limitations
- May not accurately capture nuanced sentiments such as sarcasm.
- Limited by the quality and variety of the dataset.

Overall, the sentiment analysis model provides a useful tool for understanding customer sentiments in product reviews but should be used in conjunction with other methods for comprehensive analysis.

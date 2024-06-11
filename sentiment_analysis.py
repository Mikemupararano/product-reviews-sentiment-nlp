# Importing important libraries
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from spacytextblob.spacytextblob import SpacyTextBlob

# Downloading necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Load the spaCy model and add the TextBlob component
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

try:
    # Load the dataset and handle mixed types
    df = pd.read_csv('amazon_product_reviews.csv', dtype=str)

    # Select the 'reviews.title' column and drop missing values
    reviews_data = df['reviews.title'].dropna().reset_index(drop=True)

    # Preprocess the text data: remove stopwords and perform basic text cleaning
    def preprocess_text(text):
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
        return ' '.join(tokens)

    clean_data = reviews_data.apply(preprocess_text)

    # Define a function for sentiment analysis using polarity and sentiment attributes
    def analyze_sentiment(review):
        doc = nlp(review)
        polarity = doc._.blob.polarity
        sentiment = doc._.blob.sentiment
        if polarity > 0:
            return 'Positive', polarity, sentiment
        elif polarity < 0:
            return 'Negative', polarity, sentiment
        else:
            return 'Neutral', polarity, sentiment

    # Test the sentiment analysis function on sample reviews
    sample_reviews = clean_data.sample(5)
    sample_results = sample_reviews.apply(analyze_sentiment)

    # Print sample reviews and their sentiments
    for review, result in zip(sample_reviews, sample_results):
        sentiment, polarity, sentiment_score = result
        print(f"Review: {review}\nSentiment: {sentiment}\nPolarity: {polarity}\nSentiment Score: {sentiment_score}\n")

    # Generate a brief report
    report = f"""
    Sentiment Analysis Report
    =========================
    1. Description of the dataset:
       - The dataset contains consumer reviews of Amazon products.
       - The 'reviews.title' column represents the product review titles used for sentiment analysis.

    2. Preprocessing steps:
       - Removed missing values.
       - Removed stopwords.
       - Converted text to lowercase and removed non-alphabetic characters.

    3. Evaluation of results:
       - Sample reviews were analyzed, and their sentiments were predicted as positive, negative, or neutral based on polarity scores.

    4. Insights:
       - The model can classify the sentiment of product review titles based on their polarity scores.
       - Strengths: Efficient preprocessing and sentiment analysis using spaCy and TextBlob.
       - Limitations: Further tuning and a more complex model may be required for nuanced sentiment detection.
    """

    # Save the report to a text file
    with open('sentiment_analysis_report.txt', 'w') as f:
        f.write(report)

    # Visualization (Optional): Plotting the distribution of sentiments
    sentiments = clean_data.apply(lambda x: analyze_sentiment(x)[0])
    sentiment_counts = sentiments.value_counts()

    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar')
    plt.title('Sentiment Distribution of Product Review Titles')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")

# I have used the following resource to write some of the code:
# https://github.com/PacktPublishing/Python-Data-Analysis-Third-Edition/tree/master/Chapter12

import sentiment_model


# Define the function to predict sentiment
def predict_sentiment(comment):
    sentiment, scores = sentiment_model.predict_sentiment(comment)
    return sentiment, scores

def run_sentiment_analysis(df, text_column):
    # Create new columns for sentiment and the scores
    df['sentiment'], df['positive_sentiment'], df['neutral_sentiment'], df['negative_sentiment'] = zip(*df[text_column].apply(
        lambda comment: process_sentiment(comment)
    ))
    # Calculate net sentiment and add it as a new column
    df['net_sentiment'] = (df['positive_sentiment'] - df['negative_sentiment']) / (1-df['neutral_sentiment'])
    
    return df
    # Helper function to unpack sentiment and scores
def process_sentiment(comment):
    sentiment, scores = predict_sentiment(comment)
    # Unpack scores (assuming [negative, neutral, positive] order)
    negative_score, neutral_score, positive_score = scores
    return sentiment, positive_score, neutral_score, negative_score

def csv_sentiment(sentiment_df, text_column):
    final_df = run_sentiment_analysis(sentiment_df, text_column)
    final_df.to_csv('final_sentiment_manual.csv')


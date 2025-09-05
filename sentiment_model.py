from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", clean_up_tokenization_spaces=True)
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

def predict_sentiment(text):
    # Encode the text using the tokenizer
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    # Get predictions
    with torch.no_grad():
        output = model(**encoded_input)
    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
    # Get the highest probability index
    top_pred = probabilities.argmax().item()

    # Mapping the indices to labels
    labels = ['negative', 'neutral', 'positive']
    sentiment = labels[top_pred]
    return sentiment, probabilities.numpy().tolist()[0]  # Return sentiment and probabilities
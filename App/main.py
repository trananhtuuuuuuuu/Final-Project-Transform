import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./distilbert-sentiment')
tokenizer = DistilBertTokenizer.from_pretrained('./distilbert-sentiment')

# Streamlit app
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review, and I'll predict if it's positive or negative!")

# Input text
review = st.text_area("Movie Review:", "Type your review here...")

if st.button("Predict"):
    if review.strip() == "":
        st.error("Please enter a review!")
    else:
        # Tokenize input
        inputs = tokenizer(review, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        # Display result
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.success(f"Sentiment: **{sentiment}**")

# Instructions
st.markdown("""
### How to Use:
1. Type a movie review in the text box.
2. Click **Predict** to see if the review is positive or negative.
3. Example: "This movie was amazing!" → Positive
""")